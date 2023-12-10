import heapq

def dijkstra(graph, start, end):
    # Initialize distances as infinity and the path as empty
    distances = {vertex: float('infinity') for vertex in graph}
    previous_vertices = {vertex: None for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        # If the current vertex is the end, we can stop
        if current_vertex == end:
            break

        # Check adjacent vertices
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_vertices[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous_vertices

def shortest_path(graph, start, end):
    distances, previous_vertices = dijkstra(graph, start, end)
    path, current_vertex = [], end

    # Reconstruct the path if it exists
    if distances[end] != float('infinity'):
        while previous_vertices[current_vertex] is not None:
            path.insert(0, current_vertex)
            current_vertex = previous_vertices[current_vertex]
        path.insert(0, start)

    return distances[end], path


def read_graph_from_input():
    N, M = map(int, input("Enter number of cities and roads (N M): ").split())
    s, h = map(int, input("Enter your current city and home city (s h): ").split())
    graph = {i: {} for i in range(N)}

    print("Enter the roads in the format 'u v w':")
    for _ in range(M):
        u, v, w = map(int, input().split())
        graph[u][v] = w
        graph[v][u] = w

    return graph, s, h

graph, s, h = read_graph_from_input()
weight, path = shortest_path(graph, s, h)

if weight != float('infinity'):
    print(f"Weight of the optimal path: {weight}")
    print(f"Number of vertices in the path: {len(path)}")
    print("Vertices in the optimal path:", ' '.join(map(str, path)))
else:
    print("-1")
