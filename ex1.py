graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('E', 1)],
    'D': [('G', 5)],
    'E': [('G', 2)],
    'G': []
}

h = {'A': 5, 'B': 4, 'C': 2, 'D': 5, 'E': 1, 'G': 0}

nodes = [[5, 0, 'A']]
parent = {'A': None}
cost = {'A': 0}
visited = []

while nodes:
    nodes.sort()
    current = nodes.pop(0)

    total_cost = current[0]
    actual_cost = current[1]
    node = current[2]

    if node in visited:
        continue

    visited.append(node)
    print("Visited:", node)

    if node == 'G':
        break

    for neighbour, weight in graph[node]:
        new_cost = actual_cost + weight

        if neighbour not in cost or new_cost < cost[neighbour]:
            cost[neighbour] = new_cost
            parent[neighbour] = node
            nodes.append([new_cost + h[neighbour], new_cost, neighbour])

if 'G' in visited:
    path = []
    node = 'G'

    while node is not None:
        path.append(node)
        node = parent[node]

    path.reverse()
    print("Path:", path)
    print("Total Cost:", cost['G'])
else:
    print("Goal not found")