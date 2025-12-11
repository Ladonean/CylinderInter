import streamlit as st
import numpy as np
import trimesh
from trimesh.transformations import euler_matrix, translation_matrix
import plotly.graph_objects as go


# =============================
# Funkcje geometryczne
# =============================

def create_cylinder(radius, height, sections=64):
    """
    Tworzy walec o zadanym promieniu i wysokości, wzdłuż osi Z,
    środkiem w (0, 0, 0). Wysokość: od -h/2 do +h/2.
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

    # rotacja w kolejności X -> Y -> Z (tzw. 'rxyz')
    R = euler_matrix(rx, ry, rz, 'rxyz')
    T = translation_matrix([tx, ty, tz])

    # najpierw obrót, potem przesunięcie
    M = T @ R
    return M


def intersect_cylinders(
    r1, h1,
    r2, h2,
    rx_deg, ry_deg, rz_deg,
    tx=0.0, ty=0.0, tz=0.0,
    sections=128
):
    """
    Zwraca:
      - mesh przecięcia (trimesh.Trimesh)
      - unikalne wierzchołki przecięcia jako numpy.ndarray (N, 3)
      - oba cylindry (cyl1, cyl2) po ustawieniach
    """
    # Cylinder 1 – prosty, oś Z
    cyl1 = create_cylinder(r1, h1, sections=sections)

    # Cylinder 2 – początkowo też oś Z
    cyl2 = create_cylinder(r2, h2, sections=sections)

    # Transformacja cylindra 2: rotacja + przesunięcie
    M = euler_transform(rx_deg, ry_deg, rz_deg, tx, ty, tz)
    cyl2 = cyl2.copy()
    cyl2.apply_transform(M)

    # Boolean intersection
    # W razie problemów można spróbować:
    # inter = cyl1.intersection(cyl2, engine="scad")
    inter = cyl1.intersection(cyl2)

    if inter.is_empty:
        points_unique = np.empty((0, 3))
    else:
        points = inter.vertices
        points_unique = np.unique(points, axis=0)

    return inter, points_unique, cyl1, cyl2


def mesh_to_plotly(mesh, name, opacity=0.3, color="blue"):
    """
    Konwersja trimesh.Trimesh do obiektu Plotly Mesh3d.
    """
    if mesh.is_empty:
        return None

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
        color=color
    )


# =============================
# Aplikacja Streamlit
# =============================

def main():
    st.set_page_config(
        page_title="Przecięcie dwóch cylindrów",
        layout="wide"
    )

    st.title("Przecięcie dwóch cylindrów (trimesh + Streamlit)")

    st.markdown(
        """
        - **Cylinder 1** – zawsze prosty, wzdłuż osi Z, zadanego promienia i wysokości.  
        - **Cylinder 2** – ma własny promień, wysokość, 3 kąty (Euler X/Y/Z) oraz przesunięcie (tx, ty, tz).  
        - Wynik: **chmura punktów z przecięcia** + wizualizacja 3D.
        """
    )

    # --- Panel parametrów ---
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
    sections = st.sidebar.slider(
        "Gęstość siatki (liczba sekcji)",
        16, 256, 128, 16,
        help="Im więcej sekcji, tym dokładniejsze bryły, ale wolniejsze obliczenia."
    )

    st.markdown("Kliknij przycisk, żeby policzyć przecięcie cylindrów dla zadanych parametrów.")

    if st.button("Oblicz przecięcie"):
        with st.spinner("Liczenie przecięcia cylindrów..."):
            inter_mesh, points, cyl1, cyl2 = intersect_cylinders(
                r1, h1,
                r2, h2,
                rx, ry, rz,
                tx, ty, tz,
                sections=sections
            )

        # --- Wyniki tekstowe / tabelaryczne ---
        st.subheader("Wyniki")
        st.write(f"**Liczba punktów na przecięciu:** {points.shape[0]}")

        if points.shape[0] == 0:
            st.warning("Brak przecięcia cylindrów dla zadanych parametrów.")
        else:
            st.write("Przykładowe punkty przecięcia (pierwsze 10):")
            st.dataframe(points[:10], use_container_width=True)

            # Przygotowanie CSV do pobrania
            csv_lines = ["x,y,z"] + [f"{p[0]},{p[1]},{p[2]}" for p in points]
            csv_data = "\n".join(csv_lines)

            st.download_button(
                label="Pobierz wszystkie punkty przecięcia jako CSV",
                data=csv_data,
                file_name="intersekcja_cylindrow.csv",
                mime="text/csv"
            )

        # --- Wizualizacja 3D ---
        st.subheader("Wizualizacja 3D")

        fig = go.Figure()

        # Cylinder 1 – niebieski
        m1 = mesh_to_plotly(cyl1, "Cylinder 1", opacity=0.2, color="blue")
        # Cylinder 2 – zielony
        m2 = mesh_to_plotly(cyl2, "Cylinder 2", opacity=0.2, color="green")
        # Przecięcie – czerwone
        mi = mesh_to_plotly(inter_mesh, "Przecięcie", opacity=0.9, color="red")

        if m1:
            fig.add_trace(m1)
        if m2:
            fig.add_trace(m2)
        if mi:
            fig.add_trace(mi)

        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data"
            ),
            legend=dict(x=0.02, y=0.98)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ustaw parametry po lewej i kliknij **Oblicz przecięcie**.")


if __name__ == "__main__":
    main()
