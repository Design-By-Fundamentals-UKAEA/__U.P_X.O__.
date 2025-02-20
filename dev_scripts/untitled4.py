# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 23:06:54 2024

@author: rg5749
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.algorithms import community

def create_grain_network_nx(neighbor_dict):
    """Creates a networkx graph from the entire neighbor_dict."""
    G = nx.Graph()
    G.add_edges_from([(gid, neighbor) for gid, neighbors in neighbor_dict.items() for neighbor in neighbors])
    return G


G1 = create_grain_network_nx(tgt_nneighgids)
G2 = create_grain_network_nx(smp_nneighgids)


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
