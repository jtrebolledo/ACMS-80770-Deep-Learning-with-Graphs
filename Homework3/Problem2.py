#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 2: Programming assignment
Problem 2
"""
import torch

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch import nn
from torch.autograd.functional import jacobian
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

torch.manual_seed(0)


from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math

def IdentityTensor(adjacency_m):

    try:
        m,n,p = adjacency_m.shape
    except:
        m,n = adjacency_m.shape

    identity_tensor = []

    identity = np.eye(n).astype(float)

    return torch.tensor(identity)

def d_hat(adjency_tensor):

    # Get shape of tensor
    try:
        m,n,p = adjency_tensor.shape
    except:
        m,n = adjency_tensor.shape
    
    # Get identity tensor from adjency tensor
    identity = IdentityTensor(adjency_tensor)

    # Calculate A_hat = A + I
    adj_hat = adjency_tensor + identity

    D_t = []

    # Compute D^0.
    int_adjs = adj_hat

        # Get degree of adjency matrix
    D_t_int = np.diag(np.power(np.array(int_adjs.sum(1)), -0.5).flatten())
    
    # Return D^0.5 
    return torch.tensor(D_t_int)


def A_norm(adjs, d_hat):
    
    # Calcualte dot product left size
    #Tensor_l_DotProd = torch.matmul(adjs, d_hat)
    Tensor_l_DotProd = torch.matmul(d_hat,adjs)

    # Complete Normalization by multiply by dhat again in the other side:
    # Tensor_l_DotProd = torch.transpose(Tensor_l_DotProd,1,2)
    Anorm = torch.matmul(Tensor_l_DotProd, d_hat)
    
    return Anorm

class GCN(Module):
    """
        Graph convolutional layer
    """
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        # -- initialize weight
        #pass
        self.in_features = in_features
        self.out_features = out_features

        # Generate Weight tensor
        sd = np.sqrt(100.0 / (in_features + out_features))
        self.W = np.random.uniform(-sd, sd, size=(in_features, out_features))
        self.W = Parameter(torch.tensor(self.W))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 10. / math.sqrt(self.W.size(1))
        self.W.requires_grad = True

    def __call__(self, A, H, activation_flag):
        # -- GCN propagation rule
        # Calculate the update and aggregate step
        ## First dot product
        
        Int_DotProd = torch.matmul(H, self.W)

        Ht_plus1 = torch.matmul(A, Int_DotProd)

        if activation_flag == 1:
            # Define Activation Function
            activation =  nn.ReLU()
            # Apply Activation function
            H_output = activation(Ht_plus1)
        else:
            H_output = Ht_plus1

        return H_output


class MyModel(nn.Module):
    """
        model
    """
    def __init__(self, A, IdModel):
        super(MyModel, self).__init__()
        self.IdModel = IdModel

        if IdModel == 1:
            self.Layer1  = GCN(200, 100)
            self.Dense   = GCN(100, 1)
        elif IdModel == 2:
            self.Layer1  = GCN(200, 100)
            self.Layer2  = GCN(100, 50)
            self.Dense   = GCN(50, 1)
        elif IdModel == 3:
            self.Layer1  = GCN(200, 100)
            self.Layer2  = GCN(100, 50)
            self.Layer3  = GCN(50, 20)
            self.Layer4  = GCN(20, 20)
            self.Dense   = GCN(20, 1)

        self.A = A
        identity = IdentityTensor(self.A)

        # Calculate A_hat = A + I
        adj_hat = self.A + identity
        d_t   = d_hat(self.A)
        A_normalize      = A_norm(adj_hat, d_t)
        self.A_normalize = A_normalize
        

    def forward(self, h0):
        
        if self.IdModel == 1:
            H_Output = self.Layer1(self.A_normalize, h0, 1)
            # H_Output = self.Dense(self.A_normalize, H_Output, 1)
            

        if self.IdModel == 2:
            H_1      = self.Layer1(self.A_normalize, h0, 1)
            H_Output = self.Layer2(self.A_normalize, H_1, 1)
            # H_Output = self.Dense(self.A_normalize, H_Output, 1)

        if self.IdModel == 3:
            H_1      = self.Layer1(self.A_normalize, h0, 1)
            H_2      = self.Layer2(self.A_normalize, H_1, 1)
            H_3      = self.Layer3(self.A_normalize, H_2, 1)
            H_Output = self.Layer4(self.A_normalize, H_3, 1)
            # H_Output = self.Dense(self.A_normalize, H_Output, 1)

        

        return H_Output


"""
    Effective range
"""
# -- Initialize graph
seed = 32
n_V = 200   # total number of nodes
i = 17      # node ID
k = 0       # k-hop
G = nx.barabasi_albert_graph(n_V, 2, seed=seed)

# -- plot graph
layout = nx.spring_layout(G, seed=seed, iterations=400)
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)

# -- plot neighborhood
nodes = nx.single_source_shortest_path_length(G,i, cutoff=k)
im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

# -- visualize
plt.colorbar(im2)
plt.title("Node 17 with K = 0")
plt.show()
plt.close()

nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
# -- plot neighborhood
nodes = nx.single_source_shortest_path_length(G,i, cutoff=2)
im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

# -- visualize
plt.colorbar(im2)
plt.title("Node 17 with K = 2")
plt.show()
plt.close()

nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
# -- plot neighborhood
nodes = nx.single_source_shortest_path_length(G,i, cutoff=4)
im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

# -- visualize
plt.colorbar(im2)
plt.title("Node 17 with K = 4")
plt.show()
plt.close()

nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
# -- plot neighborhood
nodes = nx.single_source_shortest_path_length(G,i, cutoff=6)
im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

# -- visualize
plt.colorbar(im2)
plt.title("Node 17 with K = 6")
plt.show()
plt.close()




nodeId = 27
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
nodes = nx.single_source_shortest_path_length(G,nodeId, cutoff=0)
im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

# -- visualize
plt.colorbar(im2)
plt.title("Node 27 with K = 0")
plt.show()
plt.close()

nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
nodes = nx.single_source_shortest_path_length(G,nodeId, cutoff=2)
im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

# -- visualize
plt.colorbar(im2)
plt.title("Node 27 with K = 2")
plt.show()
plt.close()

nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
# -- plot neighborhood
nodes = nx.single_source_shortest_path_length(G,nodeId, cutoff=4)
im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

# -- visualize
plt.colorbar(im2)
plt.title("Node 27 with K = 4")
plt.show()
plt.close()

nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
# -- plot neighborhood
nodes = nx.single_source_shortest_path_length(G,nodeId, cutoff=6)
im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

# -- visualize
plt.colorbar(im2)
plt.title("Node 27 with K = 6")
plt.show()
plt.close()

#obtain the adjacency matrix (A)
A = nx.to_numpy_matrix(G, range(0,n_V)) #nx.adjacency_matrix(G)
# print(A[0])
labels = range(0,n_V)



def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels, label_encoder.classes_

labels_encoded, classes = encode_label(labels)

# print(labels_encoded)      # Encoding labels
# print(classes)             # Decoding Label

# """
#     Influence score
# """
# -- Initialize the model and node feature vectors
# model = ?
# H = ?

# -- Influence sore
# inf_score = ?

# -- plot influence scores


# In[8]:


A_tensor = torch.tensor(A, dtype=torch.double)
H_tensor = torch.tensor(labels_encoded, dtype=torch.double, requires_grad=True)

identity = IdentityTensor(A_tensor)
adj_hat = A_tensor + identity
d_t   = d_hat(A_tensor)
A_normalize = A_norm(adj_hat, d_t)
A_inv = torch.inverse(A_normalize)
H_inv = torch.inverse(H_tensor)


model1 = MyModel(A_tensor, 1)
model2 = MyModel(A_tensor, 2)
model3 = MyModel(A_tensor, 3)

H_out_1 = model1(H_tensor)
H_out_2 = model2(H_tensor)
H_out_3 = model3(H_tensor)


Jacobian_1 = jacobian(model1,H_tensor)
Jacobian_2 = jacobian(model2,H_tensor)
Jacobian_3 = jacobian(model3,H_tensor)


# In[10]:


def Influence_Score(Node, Jacobian_Matrix):
    I_score = []
    for j in range(200):
        J = Jacobian_Matrix[Node,:,j:]
        I_score.append(J.sum())
    
    return I_score


# In[11]:


nodeId = 17
Jacobian_1_aux = Influence_Score(nodeId, Jacobian_1)  #Jacobian_1[nodeId,:,:,:].sum(2).sum(0)
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=50)
nodes = nx.single_source_shortest_path_length(G,nodeId, cutoff=0)
im2 = nx.draw_networkx_nodes(G, pos=layout, node_color=Jacobian_1_aux, node_size=50)
im1 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=50)

# -- visualize
plt.colorbar(im2)
plt.title("Node 17 Model 1")
plt.show()
plt.close()

nodeId = 17
Jacobian_2_aux = Influence_Score(nodeId, Jacobian_2) #Jacobian_2[nodeId,:,:,:].sum(2).sum(0)
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=50)
nodes = nx.single_source_shortest_path_length(G,nodeId, cutoff=0)
im2 = nx.draw_networkx_nodes(G, pos=layout, node_color=Jacobian_2_aux, node_size=50)
im1 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=50)

# -- visualize
plt.colorbar(im2)
plt.title("Node 17 Model 2")
plt.show()
plt.close()

nodeId = 17
Jacobian_3_aux = Influence_Score(nodeId, Jacobian_3) #Jacobian_3[nodeId,:,:,:].sum(2).sum(0)
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=50)
nodes = nx.single_source_shortest_path_length(G,nodeId, cutoff=0)
im2 = nx.draw_networkx_nodes(G, pos=layout, node_color=Jacobian_3_aux, node_size=50)
im1 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=50)

# -- visualize
plt.colorbar(im2)
plt.title("Node 17 Model 3")
plt.show()
plt.close()




nodeId = 27
Jacobian_1_aux = Influence_Score(nodeId, Jacobian_1) # Jacobian_1[nodeId,:,:,:].sum(2).sum(0)
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=50)
nodes = nx.single_source_shortest_path_length(G,nodeId, cutoff=0)
im2 = nx.draw_networkx_nodes(G, pos=layout, node_color=Jacobian_1_aux, node_size=50)
im1 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=50)

# -- visualize
plt.colorbar(im2)
plt.title("Node 27 Model 1")
plt.show()
plt.close()

nodeId = 27
Jacobian_2_aux = Influence_Score(nodeId, Jacobian_2) #Jacobian_2[nodeId,:,:,:].sum(2).sum(0)
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=50)
nodes = nx.single_source_shortest_path_length(G,nodeId, cutoff=0)
im2 = nx.draw_networkx_nodes(G, pos=layout, node_color=Jacobian_2_aux, node_size=50)
im1 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=50)

# -- visualize
plt.colorbar(im2)
plt.title("Node 27 Model 2")
plt.show()
plt.close()


nodeId = 27
Jacobian_3_aux = Influence_Score(nodeId, Jacobian_3) #Jacobian_3[nodeId,:,:,:].sum(2).sum(0)
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=50)
nodes = nx.single_source_shortest_path_length(G,nodeId, cutoff=0)
im2 = nx.draw_networkx_nodes(G, pos=layout, node_color=Jacobian_3_aux, node_size=50)
im1 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=50)

# -- visualize
plt.colorbar(im2)
plt.title("Node 27 Model 3")
plt.show()
plt.close()

