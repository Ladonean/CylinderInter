import streamlit as st
import numpy as np
import pandas as pd
import trimesh
from trimesh.transformations import euler_matrix, translation_matrix, rotation_matrix
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline


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
    Tworzy siatkę cylindra (trimesh) o zadanym promieniu i wysokości,
    którego oś jest wzdłuż osi Y:
      - tworzymy cylinder wzdłuż osi Z (domyślne w trimesh),
      - obracamy o +90° wokół osi X (oś Z -> oś -Y).
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
      Parametryzacja (przed obrotem):
        x = r1 * cos(theta)
        z = r1 * sin(theta)
        y ∈ [-h1/2, h1/2]

    Cylinder 2 (w lokalnym układzie):
        x2^2 + z2^2 = r2^2, |y2| <= h2/2

    Zwraca:
      - points (N, 3)  – punkty przecięcia w globalnych współrzędnych
      - Rz_base, M2    – macierze transformacji (dla wizualizacji cylindrów)
    """
    # Macierz transformacji cylindra 2
    M2 = euler_transform(rx2_deg, ry2_deg, rz2_deg, tx2, ty2, tz2)
    M2_inv = np.linalg.inv(M2)

    # Obrót cylindra 1 wokół Z
    Rz_base = rotation_z(base_rot_z_deg)

    # Sampling powierzchni cylindra 1 (oś Y)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ys = np.linspace(-h1 / 2.0, h1 / 2.0, n_y)

    Theta, Y = np.meshgrid(thetas, ys)
    X = r1 * np.cos(Theta)
    Z = r1 * np.sin(Theta)

    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)
    N = pts.shape[0]

    # Obrót cylindra 1 wokół Z
    pts_h = np.hstack([pts, np.ones((N, 1))])     # (N, 4)
    pts_rot = (Rz_base @ pts_h.T).T              # (N, 4)
    pts_global = pts_rot[:, :3]                  # (N, 3)

    # Do lokalnego układu cylindra 2
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

    return intersection_points, Rz_base, M2


def fit_oval_and_sample(points: np.ndarray, step_deg: float = 0.5):
    """
    Z surowych punktów jednego przecięcia:
      1. Dopasuj płaszczyznę (PCA),
      2. rzutuj na tę płaszczyznę -> współrzędne 2D (u,v),
      3. ustaw środek w (0,0) (średnia),
      4. przelicz punkty na (angle, radius),
      5. dopasuj splajn 1D r(angle),
      6. zwróć punkty na owalu co 'step_deg' stopni od środka.

    Zwraca:
      - angles_deg (M,)   – kąty w stopniach (0..~360)
      - points_oval (M,3) – punkty 3D owalu
      - center (3,)       – środek owalu w 3D
    """
    if points.shape[0] < 10:
        # za mało punktów na sensowne dopasowanie
        return np.array([]), np.empty((0, 3)), np.zeros(3)

    # 1. Środek w 3D
    center = points.mean(axis=0)

    # 2. PCA – znajdź płaszczyznę
    X = points - center
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # najmniejsza wartość -> normalna, dwie największe -> baza w płaszczyźnie
    e1 = eigvecs[:, 2]   # główny kierunek
    e2 = eigvecs[:, 1]   # drugi w płaszczyźnie

    # 3. Projekcja na płaszczyznę (u,v)
    u = X @ e1
    v = X @ e2

    # 4. (angle, radius)
    angles = np.arctan2(v, u)               # [-pi, pi]
    angles = (angles + 2 * np.pi) % (2 * np.pi)  # [0, 2pi)
    radius = np.sqrt(u**2 + v**2)

    # Sortowanie po kącie
    idx = np.argsort(angles)
    angles_sorted = angles[idx]
    radius_sorted = radius[idx]

    # 5. Splajn r(angle)
    try:
        spl = UnivariateSpline(angles_sorted, radius_sorted,
                               s=len(radius_sorted) * 1e-6, k=3)
    except Exception:
        # fallback – bez splajnu, tylko posortowane punkty
        angles_deg = np.degrees(angles_sorted)
        points_oval = center + np.outer(radius_sorted * np.cos(angles_sorted), e1) \
                               + np.outer(radius_sorted * np.sin(angles_sorted), e2)
        return angles_deg, points_oval, center

    # 6. Sampling co step_deg po kącie 0..2pi
    step_rad = np.radians(step_deg)
    angles_samp = np.arange(0.0, 2.0 * np.pi + step_rad, step_rad)
    radius_samp = spl(angles_samp)

    u_samp = radius_samp * np.cos(angles_samp)
    v_samp = radius_samp * np.sin(angles_samp)

    points_oval = center + np.outer(u_samp, e1) + np.outer(v_samp, e2)
    angles_deg = np.degrees(angles_samp)

    return angles_deg, points_oval, center


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
        page_title="Przecięcie cylindrów – dwa owale co kąt",
        layout="wide"
    )

    st.title("Przecięcie dwóch cylindrów – góra/dół, owal + punkty co kąt")

    st.markdown(
        """
        - **Cylinder 1**: oś **Y**, promień \\(r_1\\), wysokość \\(h_1\\), obrót wokół osi Z.  
        - **Cylinder 2**: oś Y w lokalnym układzie, promień \\(r_2\\), wysokość \\(h_2\\),
          kąty Eulera \\(r_x, r_y, r_z\\) + przesunięcie.  
        - Z surowych punktów przecięcia:
          - dzielimy na **górne** i **dolne** przecięcie (po osi Y),
          - dla każdego dopasowujemy osobny **owal**,
          - wyznaczamy punkty co zadany kąt (np. 0,5°) od środka owalu.  
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
    st.sidebar.header("Sampling powierzchni cylindra 1")

    n_theta = st.sidebar.slider(
        "Próbki po obwodzie (n_theta)",
        72, 2880, 720, 72
    )
    n_y = st.sidebar.slider(
        "Próbki po wysokości (n_y)",
        40, 1600, 400, 40
    )
    tol_factor = st.sidebar.slider(
        "Tolerancja promienia [% min(r1, r2)]",
        0.1, 5.0, 0.5, 0.1
    ) / 100.0

    st.sidebar.write(
        f"**Łącznie próbkowanych punktów:** ~{n_theta * n_y:,}"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Owal – punkty wynikowe")
    step_deg = st.sidebar.slider(
        "Krok kątowy [°] od środka",
        0.1, 5.0, 0.5, 0.1
    )

    st.markdown("Ustaw parametry i kliknij **Oblicz owale przecięcia**.")

    if st.button("Oblicz owale przecięcia"):
        with st.spinner("Liczenie punktów przecięcia i owalów..."):
            # surowe punkty przecięcia
            points_raw, Rz_base, M2 = sample_intersection_points(
                r1, h1,
                r2, h2,
                rx2, ry2, rz2,
                tx2, ty2, tz2,
                base_rot_z_deg=base_rot_z,
                n_theta=n_theta,
                n_y=n_y,
                tol_factor=tol_factor,
            )

            # siatki cylindrów
            cyl1_mesh = create_cylinder_y_mesh(r1, h1, sections=64)
            cyl1_mesh.apply_transform(Rz_base)

            cyl2_mesh = create_cylinder_y_mesh(r2, h2, sections=64)
            cyl2_mesh.apply_transform(M2)

            # podział na górę / dół po osi Y
            if points_raw.shape[0] > 0:
                y_vals = points_raw[:, 1]
                y_med = np.median(y_vals)

                pts_top = points_raw[y_vals >= y_med]
                pts_bottom = points_raw[y_vals < y_med]

                angles_top, oval_top, center_top = fit_oval_and_sample(pts_top, step_deg=step_deg) if pts_top.shape[0] > 0 else (np.array([]), np.empty((0, 3)), np.zeros(3))
                angles_bottom, oval_bottom, center_bottom = fit_oval_and_sample(pts_bottom, step_deg=step_deg) if pts_bottom.shape[0] > 0 else (np.array([]), np.empty((0, 3)), np.zeros(3))
            else:
                pts_top = np.empty((0, 3))
                pts_bottom = np.empty((0, 3))
                angles_top = angles_bottom = np.array([])
                oval_top = oval_bottom = np.empty((0, 3))
                center_top = center_bottom = np.zeros(3)

        # ---- WYNIKI ----
        st.subheader("Wyniki")

        st.write(f"**Liczba surowych punktów przecięcia (sampling):** {points_raw.shape[0]}")
        st.write(f"**Górne przecięcie – liczba punktów surowych:** {pts_top.shape[0]}")
        st.write(f"**Dolne przecięcie – liczba punktów surowych:** {pts_bottom.shape[0]}")

        if points_raw.shape[0] == 0:
            st.warning("Brak punktów przecięcia (w granicach zadanej tolerancji i próbkowania).")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Górne przecięcie")
                if oval_top.shape[0] == 0:
                    st.info("Brak punktów owalu dla górnego przecięcia.")
                else:
                    st.write(f"Liczba punktów owalu co {step_deg}° (góra): {oval_top.shape[0]}")
                    df_top = pd.DataFrame(
                        {
                            "angle_deg_from_center": angles_top,
                            "x": oval_top[:, 0],
                            "y": oval_top[:, 1],
                            "z": oval_top[:, 2],
                        }
                    )
                    st.dataframe(df_top.head(10), use_container_width=True)

                    csv_top = "\n".join(
                        ["angle_deg_from_center,x,y,z"]
                        + [f"{ang},{p[0]},{p[1]},{p[2]}" for ang, p in zip(angles_top, oval_top)]
                    )
                    st.download_button(
                        label=f"Pobierz owal GÓRA co {step_deg}° (CSV)",
                        data=csv_top,
                        file_name="owal_przeciecia_gora.csv",
                        mime="text/csv",
                    )

            with col2:
                st.markdown("### Dolne przecięcie")
                if oval_bottom.shape[0] == 0:
                    st.info("Brak punktów owalu dla dolnego przecięcia.")
                else:
                    st.write(f"Liczba punktów owalu co {step_deg}° (dół): {oval_bottom.shape[0]}")
                    df_bottom = pd.DataFrame(
                        {
                            "angle_deg_from_center": angles_bottom,
                            "x": oval_bottom[:, 0],
                            "y": oval_bottom[:, 1],
                            "z": oval_bottom[:, 2],
                        }
                    )
                    st.dataframe(df_bottom.head(10), use_container_width=True)

                    csv_bottom = "\n".join(
                        ["angle_deg_from_center,x,y,z"]
                        + [f"{ang},{p[0]},{p[1]},{p[2]}" for ang, p in zip(angles_bottom, oval_bottom)]
                    )
                    st.download_button(
                        label=f"Pobierz owal DÓŁ co {step_deg}° (CSV)",
                        data=csv_bottom,
                        file_name="owal_przeciecia_dol.csv",
                        mime="text/csv",
                    )

        # ---- WIZUALIZACJA 3D ----
        st.subheader("Wizualizacja 3D – cylindry, surowe punkty, dwa owale")

        fig = go.Figure()

        # cylindry
        fig.add_trace(mesh_to_plotly(cyl1_mesh, "Cylinder 1", opacity=0.25, color="blue"))
        fig.add_trace(mesh_to_plotly(cyl2_mesh, "Cylinder 2", opacity=0.25, color="green"))

        # surowe punkty przecięcia
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

        # owal góra
        if oval_top.shape[0] > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=oval_top[:, 0],
                    y=oval_top[:, 1],
                    z=oval_top[:, 2],
                    mode="lines+markers",
                    name=f"Owal GÓRA co {step_deg}°",
                    marker=dict(size=3),
                )
            )

        # owal dół
        if oval_bottom.shape[0] > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=oval_bottom[:, 0],
                    y=oval_bottom[:, 1],
                    z=oval_bottom[:, 2],
                    mode="lines+markers",
                    name=f"Owal DÓŁ co {step_deg}°",
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
        st.info("Kliknij **Oblicz owale przecięcia**, żeby zobaczyć wynik.")


if __name__ == "__main__":
    main()
