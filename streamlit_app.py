import streamlit as st
import numpy as np
import trimesh
from trimesh.transformations import euler_matrix, translation_matrix, rotation_matrix
import plotly.graph_objects as go
from scipy.interpolate import splprep, splev


# =============================
# Funkcje pomocnicze – geometria
# =============================

def euler_transform(rx_deg, ry_deg, rz_deg, tx=0.0, ty=0.0, tz=0.0):
    """
    Macierz 4x4: rotacja (Euler XYZ) + translacja.
    Kąty w stopniach.
    """
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    R = euler_matrix(rx, ry, rz, 'rxyz')
    T = translation_matrix([tx, ty, tz])

    M = T @ R
    return M


def rotation_z(angle_deg: float) -> np.ndarray:
    """
    Macierz 4x4 rotacji wokół osi Z o zadany kąt w stopniach.
    """
    a = np.radians(angle_deg)
    c = np.cos(a)
    s = np.sin(a)
    R = np.array([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return R


def create_cylinder_y_mesh(radius, height, sections=64):
    """
    Tworzy siatkę cylindra o zadanym promieniu i wysokości,
    którego oś jest wzdłuż osi Y.

    Implementacja:
      - tworzymy cylinder wzdłuż osi Z (domyślne w trimesh),
      - obracamy go o +90° wokół osi X, aby oś Z -> oś -Y.
    """
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    Rx = rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    mesh.apply_transform(Rx)
    return mesh


def sample_intersection_points(
    r1, h1,
    r2, h2,
    rx2_deg, ry2_deg, rz2_deg,
    tx2=0.0, ty2=0.0, tz2=0.0,
    base_rot_z_deg=0.0,
    n_theta=720,
    n_y=400,
    tol_factor=0.005,
):
    """
    NUMERYCZNE przybliżenie krzywej przecięcia.

    Cylinder 1:
      - oś Y, promień r1, wysokość h1,
      - dodatkowy obrót wokół Z o base_rot_z_deg.
      Param (przed obrotem Z):
        x = r1 * cos(theta)
        z = r1 * sin(theta)
        y ∈ [-h1/2, h1/2]

    Cylinder 2 (w lokalnym układzie):
        x2^2 + z2^2 = r2^2, |y2| <= h2/2

    Zwraca:
      - points (N, 3)  – punkty przecięcia w globalnych współrzędnych
      - theta_deg_sel (N,) – odpowiadający kąt (w stopniach) cylindra 1
      - Rz_base, M2 – macierze transformacji do wizualizacji cylindrów
    """
    # Macierz transformacji cylindra 2
    M2 = euler_transform(rx2_deg, ry2_deg, rz2_deg, tx2, ty2, tz2)
    M2_inv = np.linalg.inv(M2)

    # Obrót cylindra 1 wokół Z (globalnie)
    Rz_base = rotation_z(base_rot_z_deg)

    # Parametry próbkowania po powierzchni cylindra 1 (oś Y)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ys = np.linspace(-h1 / 2.0, h1 / 2.0, n_y)

    Theta, Y = np.meshgrid(thetas, ys)  # (n_y, n_theta)
    X = r1 * np.cos(Theta)
    Z = r1 * np.sin(Theta)

    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)
    theta_flat = Theta.ravel()  # (N,)
    N = pts.shape[0]

    # Zastosuj obrót cylindra 1 wokół Z
    pts_h = np.hstack([pts, np.ones((N, 1))])     # (N, 4)
    pts_rot = (Rz_base @ pts_h.T).T              # (N, 4)
    pts_global = pts_rot[:, :3]                  # (N, 3)

    # Przejście do lokalnego układu cylindra 2
    pts_glob_h = np.hstack([pts_global, np.ones((N, 1))])  # (N, 4)
    pts_local2 = (M2_inv @ pts_glob_h.T).T                 # (N, 4)

    x2 = pts_local2[:, 0]
    y2 = pts_local2[:, 1]
    z2 = pts_local2[:, 2]

    rho2 = np.sqrt(x2**2 + z2**2)

    tol = tol_factor * min(r1, r2)
    on_radius = np.abs(rho2 - r2) < tol
    within_height = np.abs(y2) <= (h2 / 2.0 + tol)

    mask = on_radius & within_height

    intersection_points = pts_global[mask]
    theta_sel = theta_flat[mask]            # w radianach

    theta_deg_sel = np.degrees(theta_sel)   # dla wygody

    return intersection_points, theta_deg_sel, Rz_base, M2


def spline_on_curve(points: np.ndarray, theta_deg: np.ndarray, step_deg: float = 0.5):
    """
    Dopasowanie splajnu 3D do punktów krzywej przecięcia,
    parametryzowanej kątem theta (w stopniach) cylindra 1.

    Zwraca:
      - theta_samp_deg (M,) – kąty próbkowania co step_deg
      - curve_samp (M, 3)   – współrzędne wygładzonej krzywej
    """
    if points.shape[0] < 5:
        # Za mało punktów – zwracamy tylko posortowane oryginały
        idx = np.argsort(theta_deg)
        return theta_deg[idx], points[idx]

    # Sortowanie po kącie
    theta_rad = np.radians(theta_deg)
    idx = np.argsort(theta_rad)
    t = theta_rad[idx]
    pts = points[idx]

    # Odwinięcie (na wypadek skoku w okolicach 0/360)
    t_unwrap = np.unwrap(t)

    # Dopasowanie splajnu parametrycznego
    try:
        # małe wygładzenie s (żeby tylko "lekko" uśrednić)
        tck, u = splprep([pts[:, 0], pts[:, 1], pts[:, 2]],
                         u=t_unwrap, s=len(pts) * 1e-6, k=3)
    except Exception:
        # fallback – bez splajnu
        return theta_deg[idx], points[idx]

    # Zakres parametru
    t_min, t_max = t_unwrap[0], t_unwrap[-1]

    # krok w radianach odpowiadający step_deg
    step_rad = np.radians(step_deg)
    t_samp = np.arange(t_min, t_max + step_rad, step_rad)

    xs, ys, zs = splev(t_samp, tck)
    curve_samp = np.stack([xs, ys, zs], axis=1)
    theta_samp_deg = np.degrees(t_samp)

    return theta_samp_deg, curve_samp


def mesh_to_plotly(mesh, name, opacity=0.3, color="blue"):
    """
    Konwersja trimesh.Trimesh do Plotly Mesh3d.
    """
    x, y, z = mesh.vertices.T
    i, j, k = mesh.faces.T

    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        name=name,
        opacity=opacity,
        color=color,
    )


# =============================
# Aplikacja Streamlit
# =============================

def main():
    st.set_page_config(
        page_title="Przecięcie dwóch cylindrów – splajn",
        layout="wide"
    )

    st.title("Przecięcie dwóch cylindrów – krzywa przecięcia + splajn + punkty co 0,5°")

    st.markdown(
        """
        - Cylinder 1: oś **Y**, promień \\(r_1\\), wysokość \\(h_1\\), obrót wokół osi Z.  
        - Cylinder 2: oś Y w swoim lokalnym układzie, promień \\(r_2\\), wysokość \\(h_2\\),
          kąty Eulera \\(r_x, r_y, r_z\\) + przesunięcie \\((t_x, t_y, t_z)\\).  
        - Krzywa przecięcia:
          1. liczona numerycznie przez sampling powierzchni cylindra 1,  
          2. wygładzona splajnem 3D,  
          3. próbkowana co **0,5°** po parametrze kątowym cylindra 1.
        """
    )

    # ---- PANEL PARAMETRÓW ----

    st.sidebar.header("Cylinder 1 (bazowy, oś Y)")
    r1 = st.sidebar.slider("Promień cylindra 1 (r1)", 0.1, 5.0, 1.0, 0.1)
    h1 = st.sidebar.slider("Wysokość cylindra 1 (h1)", 0.1, 20.0, 4.0, 0.1)

    base_rot_z = st.sidebar.slider(
        "Obrót cylindra 1 wokół osi Z (stopnie)",
        -180.0, 180.0, 0.0, 1.0
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Cylinder 2 (oś Y w lokalnych współrzędnych)")

    r2 = st.sidebar.slider("Promień cylindra 2 (r2)", 0.1, 5.0, 1.0, 0.1)
    h2 = st.sidebar.slider("Wysokość cylindra 2 (h2)", 0.1, 20.0, 4.0, 0.1)

    st.sidebar.subheader("Kąty Eulera cylindra 2 (stopnie)")
    rx2 = st.sidebar.slider("rx (obrót wokół osi X)", -180.0, 180.0, 45.0, 1.0)
    ry2 = st.sidebar.slider("ry (obrót wokół osi Y)", -180.0, 180.0, 0.0, 1.0)
    rz2 = st.sidebar.slider("rz (obrót wokół osi Z)", -180.0, 180.0, 0.0, 1.0)

    st.sidebar.subheader("Przesunięcie cylindra 2")
    tx2 = st.sidebar.slider("tx", -10.0, 10.0, 0.0, 0.1)
    ty2 = st.sidebar.slider("ty", -10.0, 10.0, 0.0, 0.1)
    tz2 = st.sidebar.slider("tz", -10.0, 10.0, 0.0, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.header("Gęstość próbkowania powierzchni cylindra 1")

    n_theta = st.sidebar.slider(
        "Próbki po obwodzie cylindra 1 (n_theta)",
        72, 2880, 720, 72
    )
    n_y = st.sidebar.slider(
        "Próbki po wysokości cylindra 1 (n_y)",
        40, 1600, 400, 40
    )
    tol_factor = st.sidebar.slider(
        "Tolerancja promienia [% min(r1, r2)]",
        0.1, 5.0, 0.5, 0.1
    ) / 100.0

    st.sidebar.write(
        f"**Łącznie próbkowanych punktów powierzchni:** ~{n_theta * n_y:,}"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Splajn krzywej")
    step_deg = st.sidebar.slider(
        "Krok po kącie dla punktów wynikowych [°]",
        0.1, 5.0, 0.5, 0.1
    )

    st.markdown("Ustaw parametry i kliknij **Oblicz krzywą przecięcia**.")

    if st.button("Oblicz krzywą przecięcia"):
        with st.spinner("Liczenie krzywej przecięcia i splajnu..."):
            points_raw, theta_raw_deg, Rz_base, M2 = sample_intersection_points(
                r1, h1,
                r2, h2,
                rx2, ry2, rz2,
                tx2, ty2, tz2,
                base_rot_z_deg=base_rot_z,
                n_theta=n_theta,
                n_y=n_y,
                tol_factor=tol_factor,
            )

            # Siatki cylindrów do wizualizacji
            cyl1_mesh = create_cylinder_y_mesh(r1, h1, sections=64)
            cyl1_mesh.apply_transform(Rz_base)

            cyl2_mesh = create_cylinder_y_mesh(r2, h2, sections=64)
            cyl2_mesh.apply_transform(M2)

            # Splajn + próbkowanie co step_deg
            if points_raw.shape[0] > 0:
                theta_samp_deg, curve_samp = spline_on_curve(
                    points_raw, theta_raw_deg, step_deg=step_deg
                )
            else:
                theta_samp_deg, curve_samp = np.array([]), np.empty((0, 3))

        # ---- WYNIKI ----
        st.subheader("Wyniki")

        st.write(f"**Liczba pierwotnych punktów przecięcia (sampling):** {points_raw.shape[0]}")
        st.write(f"**Liczba punktów na splajnie co {step_deg}°:** {curve_samp.shape[0]}")

        if points_raw.shape[0] == 0:
            st.warning("Brak punktów przecięcia (w granicach zadanej tolerancji i próbkowania).")
        else:
            st.write("Przykładowe punkty splajnu (pierwsze 10):")
            import pandas as pd
            df = pd.DataFrame(
                {
                    "theta_deg": theta_samp_deg,
                    "x": curve_samp[:, 0],
                    "y": curve_samp[:, 1],
                    "z": curve_samp[:, 2],
                }
            )
            st.dataframe(df.head(10), use_container_width=True)

            # CSV do pobrania
            csv_lines = ["theta_deg,x,y,z"] + [
                f"{th},{p[0]},{p[1]},{p[2]}"
                for th, p in zip(theta_samp_deg, curve_samp)
            ]
            csv_data = "\n".join(csv_lines)

            st.download_button(
                label=f"Pobierz punkty splajnu co {step_deg}° jako CSV",
                data=csv_data,
                file_name="krzywa_przeciecia_splajn.csv",
                mime="text/csv"
            )

        # ---- WIZUALIZACJA 3D ----
        st.subheader("Wizualizacja 3D – cylindry + surowe punkty + splajn")

        fig = go.Figure()

        # Cylinder 1 – półprzezroczysty
        fig.add_trace(
            mesh_to_plotly(cyl1_mesh, "Cylinder 1", opacity=0.25, color="blue")
        )

        # Cylinder 2 – półprzezroczysty
        fig.add_trace(
            mesh_to_plotly(cyl2_mesh, "Cylinder 2", opacity=0.25, color="green")
        )

        # Surowe punkty przecięcia (sampling)
        if points_raw.shape[0] > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=points_raw[:, 0],
                    y=points_raw[:, 1],
                    z=points_raw[:, 2],
                    mode="markers",
                    name="Punkty przecięcia (surowe)",
                    marker=dict(size=2),
                )
            )

        # Splajn jako linia
        if curve_samp.shape[0] > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=curve_samp[:, 0],
                    y=curve_samp[:, 1],
                    z=curve_samp[:, 2],
                    mode="lines+markers",
                    name=f"Splajn co {step_deg}°",
                    marker=dict(size=3),
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            ),
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=0, r=0, t=30, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Kliknij **Oblicz krzywą przecięcia**, żeby zobaczyć wynik.")


if __name__ == "__main__":
    main()
