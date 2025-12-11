import streamlit as st
import numpy as np
from trimesh.transformations import euler_matrix, translation_matrix
import plotly.graph_objects as go


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


def sample_intersection_points(
    r1, h1,
    r2, h2,
    rx2_deg, ry2_deg, rz2_deg,
    tx2=0.0, ty2=0.0, tz2=0.0,
    base_rot_z_deg=0.0,
    n_theta=360,
    n_y=200,
    tol_factor=0.02,
):
    """
    NUMERYCZNE przybliżenie krzywej przecięcia:

    - Cylinder 1: oś Y, promień r1, wysokość h1.
      Parametryzacja (przed obrotem wokół Z):
        x = r1 * cos(theta)
        z = r1 * sin(theta)
        y ∈ [-h1/2, h1/2]

      Następnie cylinder 1 jest obracany wokół osi Z o base_rot_z_deg.

    - Cylinder 2: dowolna orientacja (Euler XYZ) + translacja.

    Próbkujemy powierzchnię cylindra 1 i sprawdzamy,
    które punkty leżą (z dokładnością tol) na powierzchni cylindra 2.

    Zwraca:
      - points (N, 3): punkty przybliżonej krzywej przecięcia.
    """
    # Macierz transformacji cylindra 2
    M2 = euler_transform(rx2_deg, ry2_deg, rz2_deg, tx2, ty2, tz2)
    M2_inv = np.linalg.inv(M2)

    # Obrót cylindra 1 wokół Z (globalnie)
    Rz_base = rotation_z(base_rot_z_deg)

    # Parametry próbkowania po powierzchni cylindra 1 (oś Y)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ys = np.linspace(-h1 / 2.0, h1 / 2.0, n_y)

    Theta, Y = np.meshgrid(thetas, ys)
    X = r1 * np.cos(Theta)
    Z = r1 * np.sin(Theta)

    # Punkty w globalnym układzie przed obrotem wokół Z
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)
    N = pts.shape[0]

    # Zastosuj obrót cylindra 1 wokół Z
    pts_h = np.hstack([pts, np.ones((N, 1))])  # (N, 4)
    pts_rot = (Rz_base @ pts_h.T).T  # (N, 4)
    pts_global = pts_rot[:, :3]      # (N, 3)

    # Przejście do lokalnego układu cylindra 2: p' = M2_inv * p
    pts_glob_h = np.hstack([pts_global, np.ones((N, 1))])  # (N, 4)
    pts_local2 = (M2_inv @ pts_glob_h.T).T  # (N, 4)

    x2 = pts_local2[:, 0]
    y2 = pts_local2[:, 1]
    z2 = pts_local2[:, 2]

    # Cylinder 2 w swoim lokalnym układzie współrzędnych:
    # x2^2 + z2^2 = r2^2, |y2| <= h2/2
    rho2 = np.sqrt(x2**2 + z2**2)

    # tolerancja względem promienia
    tol = tol_factor * min(r1, r2)

    on_radius = np.abs(rho2 - r2) < tol
    within_height = np.abs(y2) <= (h2 / 2.0 + tol)

    mask = on_radius & within_height
    intersection_points = pts_global[mask]

    return intersection_points


# =============================
# Aplikacja Streamlit
# =============================

def main():
    st.set_page_config(
        page_title="Przecięcie dwóch cylindrów – punkty",
        layout="wide"
    )

    st.title("Przecięcie dwóch cylindrów – tylko punkty przecięcia")

    st.markdown(
        """
        - **Cylinder 1 (bazowy)** – oś **Y**, promień \\(r_1\\), wysokość \\(h_1\\), 
          dodatkowy **obrót wokół osi Z**.  
        - **Cylinder 2** – promień \\(r_2\\), wysokość \\(h_2\\), 
          trzy kąty Eulera (X, Y, Z) + przesunięcie (tx, ty, tz).  
        - Wynik: tylko **punkty przybliżonej krzywej przecięcia** (sampling powierzchni cylindra 1).
        """
    )

    # ---- PANEL PARAMETRÓW ----
    st.sidebar.header("Parametry cylindra bazowego (oś Y)")

    r1 = st.sidebar.slider("Promień cylindra 1 (r1)", 0.1, 5.0, 1.0, 0.1)
    h1 = st.sidebar.slider("Wysokość cylindra 1 (h1)", 0.1, 20.0, 4.0, 0.1)

    base_rot_z = st.sidebar.slider(
        "Obrót cylindra 1 wokół osi Z (stopnie)",
        -180.0, 180.0, 0.0, 1.0
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Parametry cylindra 2")

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
    st.sidebar.header("Gęstość próbkowania")

    n_theta = st.sidebar.slider(
        "Próbki po obwodzie cylindra 1 (n_theta)",
        36, 1440, 360, 12
    )
    n_y = st.sidebar.slider(
        "Próbki po wysokości cylindra 1 (n_y)",
        20, 800, 200, 10
    )
    tol_factor = st.sidebar.slider(
        "Tolerancja promienia [% min(r1, r2)]",
        0.1, 10.0, 2.0, 0.1
    ) / 100.0

    st.sidebar.write(
        f"**Łącznie próbkowanych punktów:** ~{n_theta * n_y:,}"
    )

    st.markdown("Ustaw parametry i kliknij **Oblicz punkty przecięcia**.")

    if st.button("Oblicz punkty przecięcia"):
        with st.spinner("Liczenie punktów przecięcia..."):
            points = sample_intersection_points(
                r1, h1,
                r2, h2,
                rx2, ry2, rz2,
                tx2, ty2, tz2,
                base_rot_z_deg=base_rot_z,
                n_theta=n_theta,
                n_y=n_y,
                tol_factor=tol_factor,
            )

        st.subheader("Wyniki")

        st.write(f"**Liczba znalezionych punktów przecięcia:** {points.shape[0]}")

        if points.shape[0] == 0:
            st.warning("Brak punktów przecięcia (w granicach zadanej tolerancji i próbkowania).")
        else:
            st.write("Przykładowe punkty (pierwsze 10):")
            st.dataframe(points[:10], use_container_width=True)

            # CSV do pobrania
            csv_lines = ["x,y,z"] + [f"{p[0]},{p[1]},{p[2]}" for p in points]
            csv_data = "\n".join(csv_lines)

            st.download_button(
                label="Pobierz wszystkie punkty przecięcia jako CSV",
                data=csv_data,
                file_name="punkty_przeciecia_cylindrow.csv",
                mime="text/csv"
            )

            # Prosta wizualizacja – tylko punkty
            st.subheader("Wizualizacja 3D (tylko punkty przecięcia)")
            fig = go.Figure()

            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    name="Punkty przecięcia",
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
                margin=dict(l=0, r=0, t=30, b=0),
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Kliknij **Oblicz punkty przecięcia**, żeby zobaczyć wynik.")


if __name__ == "__main__":
    main()
