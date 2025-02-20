# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:17:24 2024

@author: rg5749
"""
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.algorithms import community

G1 = nx.Graph()
G1.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])  # A triangle with an extra node

G2 = nx.Graph()
G2.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])  # A path of 5 nodes


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(121)
nx.draw(G1, with_labels=True)
plt.title("Network G1")
plt.subplot(122)
nx.draw(G2, with_labels=True)
plt.title("Network G2")
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
