import streamlit as st
import numpy as np
import trimesh
from trimesh.transformations import euler_matrix, translation_matrix
import plotly.graph_objects as go


# =============================
# Funkcje pomocnicze – geometria
# =============================

def create_cylinder_mesh(radius, height, sections=64):
    """
    Tworzy siatkę cylindra (trimesh) wzdłuż osi Z,
    środek w (0,0,0), z od -h/2 do +h/2.
    """
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


def euler_transform(rx_deg, ry_deg, rz_deg, tx=0.0, ty=0.0, tz=0.0):
    """
    Zwraca macierz 4x4: rotacja (Euler XYZ) + translacja.
    Kąty w stopniach.
    """
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    R = euler_matrix(rx, ry, rz, 'rxyz')
    T = translation_matrix([tx, ty, tz])

    M = T @ R
    return M


def sample_intersection_points(
    r1, h1,
    r2, h2,
    rx_deg, ry_deg, rz_deg,
    tx=0.0, ty=0.0, tz=0.0,
    n_theta=360,
    n_z=200,
    tol_factor=0.02,
):
    """
    NUMERYCZNE przybliżenie krzywej przecięcia:
    - próbkujemy powierzchnię cylindra 1 (oś Z, promień r1, wysokość h1),
    - dla każdego punktu sprawdzamy, czy leży też na powierzchni cylindra 2.

    Zwraca:
      - points (N, 3): punkty przybliżonej krzywej przecięcia
      - M: macierz transformacji cylindra 2 (do wizualizacji)
    """
    # Macierz transformacji cylindra 2
    M = euler_transform(rx_deg, ry_deg, rz_deg, tx, ty, tz)
    M_inv = np.linalg.inv(M)

    # Parametry próbkowania cylindra 1
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(-h1 / 2.0, h1 / 2.0, n_z)

    Theta, Z = np.meshgrid(thetas, zs)
    X = r1 * np.cos(Theta)
    Y = r1 * np.sin(Theta)

    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)
    N = pts.shape[0]

    # Zamiana na współrzędne lokalne cylindra 2: p' = M_inv * p
    pts_h = np.hstack([pts, np.ones((N, 1))])  # (N, 4)
    pts_local = (M_inv @ pts_h.T).T  # (N, 4)
    x2 = pts_local[:, 0]
    y2 = pts_local[:, 1]
    z2 = pts_local[:, 2]

    # Warunek "w pobliżu powierzchni cylindra 2"
    # promień w lokalnych współrzędnych cylindra 2:
    rho2 = np.sqrt(x2**2 + y2**2)

    # tolerancja względem promienia
    tol = tol_factor * min(r1, r2)

    on_radius = np.abs(rho2 - r2) < tol
    within_height = np.abs(z2) <= (h2 / 2.0 + tol)

    mask = on_radius & within_height
    intersection_points = pts[mask]

    return intersection_points, M


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
        page_title="Przecięcie dwóch cylindrów",
        layout="wide"
    )

    st.title("Przecięcie dwóch cylindrów (bez boolean, działa na Streamlit Cloud)")

    st.markdown(
        """
        - **Cylinder 1** – bazowy, prosty wzdłuż osi Z.  
        - **Cylinder 2** – ma własny promień, wysokość, 3 kąty Eulera oraz przesunięcie.  
        - Przecięcie jest liczone **numerycznie** przez próbkowanie powierzchni cylindra 1.  
        """
    )

    # ---- PANEL PARAMETRÓW ----
    st.sidebar.header("Parametry cylindrów")

    # Cylinder 1
    st.sidebar.subheader("Cylinder 1 (bazowy)")
    r1 = st.sidebar.slider("Promień cylindra 1", 0.1, 5.0, 1.0, 0.1)
    h1 = st.sidebar.slider("Wysokość cylindra 1", 0.1, 20.0, 4.0, 0.1)

    # Cylinder 2
    st.sidebar.subheader("Cylinder 2 (obracany)")
    r2 = st.sidebar.slider("Promień cylindra 2", 0.1, 5.0, 1.0, 0.1)
    h2 = st.sidebar.slider("Wysokość cylindra 2", 0.1, 20.0, 4.0, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Kąty Eulera cylindra 2 (stopnie)")
    rx = st.sidebar.slider("Obrót wokół osi X (rx)", -180.0, 180.0, 45.0, 1.0)
    ry = st.sidebar.slider("Obrót wokół osi Y (ry)", -180.0, 180.0, 0.0, 1.0)
    rz = st.sidebar.slider("Obrót wokół osi Z (rz)", -180.0, 180.0, 0.0, 1.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Przesunięcie cylindra 2")
    tx = st.sidebar.slider("tx", -10.0, 10.0, 0.0, 0.1)
    ty = st.sidebar.slider("ty", -10.0, 10.0, 0.0, 0.1)
    tz = st.sidebar.slider("tz", -10.0, 10.0, 0.0, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parametry próbkowania")
    n_theta = st.sidebar.slider("Próbki po obwodzie (n_theta)", 36, 720, 360, 12)
    n_z = st.sidebar.slider("Próbki po wysokości (n_z)", 20, 400, 200, 10)
    tol_factor = st.sidebar.slider(
        "Tolerancja względem promienia (procent minimalnego promienia)",
        0.1, 10.0, 2.0, 0.1
    ) / 100.0  # konwersja z % na ułamek

    st.markdown("Ustaw parametry i kliknij **Oblicz przecięcie**.")

    if st.button("Oblicz przecięcie"):
        with st.spinner("Liczenie przybliżonej krzywej przecięcia..."):
            points, M2 = sample_intersection_points(
                r1, h1,
                r2, h2,
                rx, ry, rz,
                tx, ty, tz,
                n_theta=n_theta,
                n_z=n_z,
                tol_factor=tol_factor
            )

            # siatki do wizualizacji
            cyl1_mesh = create_cylinder_mesh(r1, h1, sections=64)
            cyl2_mesh = create_cylinder_mesh(r2, h2, sections=64)
            cyl2_mesh = cyl2_mesh.copy()
            cyl2_mesh.apply_transform(M2)

        # ---- WYNIKI ----
        st.subheader("Wyniki")

        st.write(f"**Liczba punktów na przybliżonej krzywej przecięcia:** {points.shape[0]}")

        if points.shape[0] == 0:
            st.warning("Brak przecięcia (w granicach zadanej tolerancji i próbkowania).")
        else:
            st.write("Przykładowe punkty (pierwsze 10):")
            st.dataframe(points[:10], use_container_width=True)

            # CSV do pobrania
            csv_lines = ["x,y,z"] + [f"{p[0]},{p[1]},{p[2]}" for p in points]
            csv_data = "\n".join(csv_lines)

            st.download_button(
                label="Pobierz punkty przecięcia jako CSV",
                data=csv_data,
                file_name="intersekcja_cylindrow_sampled.csv",
                mime="text/csv"
            )

        # ---- WIZUALIZACJA 3D ----
        st.subheader("Wizualizacja 3D")

        fig = go.Figure()

        m1 = mesh_to_plotly(cyl1_mesh, "Cylinder 1", opacity=0.25, color="blue")
        m2 = mesh_to_plotly(cyl2_mesh, "Cylinder 2", opacity=0.25, color="green")

        fig.add_trace(m1)
        fig.add_trace(m2)

        # punkty przecięcia jako chmura
        if points.shape[0] > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    name="Przybliżona krzywa przecięcia",
                    marker=dict(size=3)
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
        st.info("Kliknij **Oblicz przecięcie**, żeby zobaczyć wynik.")


if __name__ == "__main__":
    main()
