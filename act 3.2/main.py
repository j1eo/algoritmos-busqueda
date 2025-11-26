import streamlit as st
import collections

# ------------------------------
# BFS para encontrar un camino
# ------------------------------
def solve_maze_bfs(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    queue = collections.deque([(start, [start])])
    visited = set()
    visited.add(start)

    while queue:
        (curr_row, curr_col), path = queue.popleft()

        if (curr_row, curr_col) == end:
            return path

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_row, next_col = curr_row + dr, curr_col + dc
            
            if (
                0 <= next_row < rows and 
                0 <= next_col < cols and
                maze[next_row][next_col] == 0 and
                (next_row, next_col) not in visited
            ):
                visited.add((next_row, next_col))
                new_path = list(path)
                new_path.append((next_row, next_col))
                queue.append(((next_row, next_col), new_path))
    
    return None


# ------------------------------
# Laberinto de ejemplo
# ------------------------------
MAZE = [
    [0,1,0,0,0,0,1,0,0,0],
    [0,1,0,1,1,0,1,0,1,0],
    [0,0,0,0,1,0,0,0,1,0],
    [1,1,1,0,1,1,1,0,1,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,1,0,1,0],
    [0,0,0,0,0,0,1,0,1,0],
    [0,1,1,1,1,0,1,0,1,0],
    [0,0,0,0,1,0,0,0,1,0],
    [0,1,1,0,0,0,1,0,0,0]
]

START = (0, 0)
END = (9, 9)


# ------------------------------
# Renderizar laberinto con emojis
# ------------------------------
def render_maze(maze, path=None):
    if path is None:
        path = []

    display_maze = []
    for r_idx, row in enumerate(maze):
        display_row = []
        for c_idx, col in enumerate(row):
            if (r_idx, c_idx) == START:
                display_row.append("🚀")
            elif (r_idx, c_idx) == END:
                display_row.append("🏁")
            elif (r_idx, c_idx) in path:
                display_row.append("🔹")
            elif col == 1:
                display_row.append("⬛")
            else:
                display_row.append("⬜")
        display_maze.append("".join(display_row))
    
    st.markdown("<br>".join(display_maze), unsafe_allow_html=True)


# ------------------------------
# Interfaz Streamlit
# ------------------------------
st.title("Visualizador de Algoritmos de Búsqueda en Laberintos")
st.header("AVANCE 3.2")

st.sidebar.header("Opciones")
algorithm = st.sidebar.selectbox("Selecciona el algoritmo", ["BFS"])
solve_button = st.sidebar.button("Resolver Laberinto")

st.subheader("Laberinto actual:")
render_maze(MAZE)

if solve_button:
    if algorithm == "BFS":
        path = solve_maze_bfs(MAZE, START, END)
        if path:
            st.success("¡Camino encontrado con BFS!")
            render_maze(MAZE, path)
            st.write(f"Longitud del camino: {len(path)} pasos")
        else:
            st.error("No se encontró un camino.")
