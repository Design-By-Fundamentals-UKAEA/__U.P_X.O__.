# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:16:21 2024

@author: rg5749
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Example grid (replace with your actual data)
# Define grid dimensions
width = 20
height = 15

# Create grid coordinates using np.meshgrid
x_coords = np.arange(width)
y_coords = np.arange(height)
xx, yy = np.meshgrid(x_coords, y_coords)
coords = np.vstack((xx.ravel(), yy.ravel())).T

# Create a dictionary mapping point IDs to coordinates
points = {i: tuple(coord) for i, coord in enumerate(coords, start=1)}


G = nx.Graph()

# Add nodes with coordinates
for point_id, coords in points.items():
    G.add_node(point_id, pos=coords)

# Add edges (4-connected grid example)
for point_id, coords in points.items():
    x, y = coords
    neighbors = [
        (x-1, y), (x+1, y), (x, y-1), (x, y+1)  # Neighbors to check
    ]
    for _nx_, ny in neighbors:
        neighbor_id = [pid for pid, c in points.items() if c == (_nx_, ny)]
        if neighbor_id:
            G.add_edge(point_id, neighbor_id[0])  # Add edge if neighbor exists

# Add edges (8-connected grid)
for point_id, coords in points.items():
    x, y = coords
    neighbors = [
        (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),  # Up, down, left, right
        (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)  # Diagonals
    ]
    for _nx_, ny in neighbors:
        neighbor_id = [pid for pid, c in points.items() if c == (_nx_, ny)]
        if neighbor_id:
            # Calculate Manhattan distance for edge weight (optional)
            weight = abs(x - _nx_) + abs(y - ny)
            G.add_edge(point_id, neighbor_id[0], weight=weight)

# Example: Find shortest path from point 1 to point 9
start_node = 1
end_node = 229
path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
print(f"Shortest path: {path}")

# Visualize the grid and shortest path
pos = nx.get_node_attributes(G, 'pos')
plt.figure(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue')

# Highlight the shortest path
path_edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2.0)

plt.title("Grid with Shortest Path")
plt.show()
# Define Jaccard similarity function
def jaccard_similarity(G1, G2):
    """Calculates Jaccard similarity between the node sets of two graphs."""
    nodes1 = set(G1.nodes())
    nodes2 = set(G2.nodes())
    intersection = len(nodes1 & nodes2)
    union = len(nodes1 | nodes2)
    return intersection / union
jaccard_sim = jaccard_similarity(G1, G2)
print(f"Jaccard Similarity (Nodes): {jaccard_sim}")
