import streamlit as st
import collections
import time
import heapq

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
# DFS para encontrar un camino
# ------------------------------
def solve_maze_dfs(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    stack = [(start, [start])]
    visited = set()
    visited.add(start)

    while stack:
        (curr_row, curr_col), path = stack.pop()

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
                stack.append(((next_row, next_col), new_path))

    return None

# ------------------------------
# A* para encontrar un camino
# ------------------------------
def solve_maze_a_star(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    
    # Función heurística (distancia Manhattan)
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, end), 0, start, []))  # (f, g, posición, camino)
    visited = set()
    
    while open_list:
        _, g, (curr_row, curr_col), path = heapq.heappop(open_list)

        if (curr_row, curr_col) == end:
            return path + [(curr_row, curr_col)]

        if (curr_row, curr_col) in visited:
            continue
        
        visited.add((curr_row, curr_col))

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_row, next_col = curr_row + dr, curr_col + dc
            
            if (
                0 <= next_row < rows and 
                0 <= next_col < cols and
                maze[next_row][next_col] == 0
            ):
                heapq.heappush(open_list, (
                    g + 1 + heuristic((next_row, next_col), end), 
                    g + 1, 
                    (next_row, next_col), 
                    path + [(curr_row, curr_col)]
                ))

    return None

# ------------------------------
# Laberinto de ejemplo
# ------------------------------
MAZE = [
[1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1],
[1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1],
[1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1],
[1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1],
[1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1],
[1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1],
[1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,1],
[1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1],
[1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1],
[1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1],
[1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,1],
[1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1],
[1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1],
[1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1],
[1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1],
[1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,1],
[1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,1],
[1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1],
[1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,1],
[1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1],
[1,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1,1],
[1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,0,1,1],
[1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1],
[1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1],
[1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,1,1],
[1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
[1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1],
[1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1],
[1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1],
[1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1],
[1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,1],
[1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,1],
[1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,1],
[1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1],
[1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1]
]

START = (0, 0)
END = (36, 30)

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
algorithm = st.sidebar.selectbox("Selecciona el algoritmo", ["BFS", "DFS", "A*"])
solve_button = st.sidebar.button("Resolver Laberinto")

st.subheader("Laberinto actual:")
render_maze(MAZE)

if solve_button:
    start_time = time.time()

    if algorithm == "BFS":
        path = solve_maze_bfs(MAZE, START, END)
    elif algorithm == "DFS":
        path = solve_maze_dfs(MAZE, START, END)
    elif algorithm == "A*":
        path = solve_maze_a_star(MAZE, START, END)

    end_time = time.time()
    exec_time = (end_time - start_time) * 1000  # ms

    if path:
        st.success(f"¡Camino encontrado con {algorithm}!")
        render_maze(MAZE, path)
        st.write(f"Longitud del camino: {len(path)} pasos")
        st.write(f"Tiempo de ejecución: **{exec_time:.4f} ms**")
    else:
        st.error("No se encontró un camino.")
        st.write(f"Tiempo de ejecución: **{exec_time:.4f} ms**")
