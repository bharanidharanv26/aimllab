graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('E', 1)],
    'D': [('G', 5)],
    'E': [('G', 2)],
    'G': []
}

h = {'A': 5, 'B': 4, 'C': 2, 'D': 5, 'E': 1, 'G': 0}

def astar(start, goal):
    open_list = [[h[start], 0, start]]
    parent = {start: None}

    while open_list:
        open_list.sort()
        f, g, node = open_list.pop(0)

        if node == goal:
            path = []
            while node:
                path.append(node)
                node = parent[node]
            print("Path:", path[::-1])
            print("Cost:", g)
            return

        for next_node, cost in graph[node]:
            new_g = g + cost
            new_f = new_g + h[next_node]

            parent[next_node] = node
            open_list.append([new_f, new_g, next_node])

    print("No path found")

astar('A', 'G')