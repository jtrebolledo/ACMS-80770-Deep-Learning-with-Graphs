"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 1: Programming assignment
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy


# -- Initialize graphs
seed = 30
G = nx.florentine_families_graph()
nodes = G.nodes()

layout = nx.spring_layout(G, seed=seed)

# -- compute jaccard's similarity
"""
    This example is using NetwrokX's native implementation to compute similarities.
    Write a code to compute Jaccard's similarity and replace with this function.
"""

def Get_Neighbors(Node, edges):
    
    Neighbors = []

    # Determine the neigbors of Node
    for w in edges:
        if w[0] == Node:
            Neighbors.append(w[1])
        if w[1] == Node:
            Neighbors.append(w[0])
    
    # Ensure a unique values in the list
    Neighbors = list(set(Neighbors))

    return Neighbors


def Get_Jaccard_Matrix(G):

    # Obtain edges from G
    edges = G.edges()

    # Obtain nodes from G
    nodes = G.nodes()

    # Storage nodes in a list
    LabelNodes = []

    for i in nodes:
        LabelNodes.append(i)


    coeff_value = []

    for i in LabelNodes:
        for j in LabelNodes:
            if i != j:
                # Get a list with neigbors of node i and j
                Neighbors_i = Get_Neighbors(i, edges) #[]
                Neighbors_j = Get_Neighbors(j, edges) #[]

                # Obtain a list with the intersection between the neighbors of node i and j
                Intersection_ij = list(set(Neighbors_i) & set(Neighbors_j))
                
                # Obtain a list with the union between the neighbors of node i and j
                Union_ij        = list(set(Neighbors_i) | set(Neighbors_j))

                # Calculate the Jaccard Index
                IndexValue      = len(list(set(Intersection_ij)))/len(list(set(Union_ij)))
                
                # Storage result in a list
                coeff_value.append([i, j, IndexValue])
    
    return coeff_value

# Get Jaccard Matrix
pred = Get_Jaccard_Matrix(G)


# -- keep a copy of edges in the graph
old_edges = copy.deepcopy(G.edges())

# -- add new edges representing similarities.
new_edges, metric = [], []
for u, v, p in pred:
    G.add_edge(u, v)
    print(f"({u}, {v}) -> {p:.8f}")
    new_edges.append((u, v))
    metric.append(p)


# -- plot Florentine Families graph
nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
nx.draw_networkx_edges(G, edgelist=old_edges, pos=layout, edge_color='gray', width=4)

# -- plot edges representing similarity
"""
    This example is randomly plotting similarities between 8 pairs of nodes in the graph. 
    Identify the ”Ginori”
"""

# By construction, I calculate the Jaccard Matrix for the all possible combination
# So it is possible obtain all Jaccarb Index using the following loop
Index_Ginori = []
for i, T in enumerate(new_edges):
    if T[0] == "Ginori":
        Index_Ginori.append(i)



ne = nx.draw_networkx_edges(G, edgelist=new_edges[Index_Ginori[0]:(Index_Ginori[-1]+1)], pos=layout, 
                            edge_color=np.asarray(metric[Index_Ginori[0]:(Index_Ginori[-1]+1)]), width=4, alpha=0.7)
plt.colorbar(ne)
plt.axis('off')
plt.show()