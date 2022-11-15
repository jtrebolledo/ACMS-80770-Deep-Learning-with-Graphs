#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 2: Programming assignment
Problem 1
"""
import torch
from torch import nn
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=UserWarning)
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor
import tensorflow as tf
from torch.nn.parameter import Parameter
import math

"""
    load data
"""
dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True,
                                           target_index=np.random.choice(range(133000), 6000, False))

V = 9
atom_types = [6, 8, 7, 9, 1]

def adj(x):
    x = x[1]
    adjacency = np.zeros((V, V)).astype(float)
    adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
    return torch.tensor(adjacency)


def sig(x):
    x = x[0]
    atoms = np.ones((V)).astype(float)
    atoms[:len(x)] = x
    out = np.array([int(atom == atom_type) for atom_type in atom_types for atom in atoms]).astype(float)
    return torch.tensor(out).reshape(5, len(atoms)).T


def target(x):
    x = x[2]
    return torch.tensor(x)


adjs = torch.stack(list(map(adj, dataset)))
sigs = torch.stack(list(map(sig, dataset)))
prop = torch.stack(list(map(target, dataset)))[:, 5]


def IdentityTensor(adjacency_m):

    try:
        m,n,p = adjacency_m.shape
    except:
        m,n = adjacency_m.shape

    identity_tensor = []

    identity = np.eye(n).astype(float)

    for i in range(m):
        identity_tensor.append(identity)

    return torch.tensor(identity_tensor)

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
    for i in range(m):
        # Get internal adjency matrix
        int_adjs = adj_hat[i]

        # Get degree of adjency matrix
        D_t_int = np.diag(np.power(np.array(int_adjs.sum(1)), -0.5).flatten())

        D_t.append(D_t_int)
    
    # Return D^0.5 
    return torch.tensor(D_t)


def A_norm(adjs, d_hat):
    
    # Calcualte dot product left size
    #Tensor_l_DotProd = torch.matmul(adjs, d_hat)
    Tensor_l_DotProd = torch.matmul(d_hat,adjs)

    # Complete Normalization by multiply by dhat again in the other side:
    # Tensor_l_DotProd = torch.transpose(Tensor_l_DotProd,1,2)
    Anorm = torch.matmul(Tensor_l_DotProd, d_hat)
    
    return Anorm

from torch.nn.modules.module import Module

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
        sd = np.sqrt(1.0 / (in_features + out_features))
        self.W = np.random.uniform(-sd, sd, size=(in_features, out_features))
        self.W = Parameter(torch.tensor(self.W))
        # self.W = Parameter(torch.FloatTensor(in_features, out_features))
        # self.W = Parameter(torch.tensor(self.W),requires_grad=True)
        #self.W = Parameter(torch.tensor(torch.FloatTensor(in_features,out_features)))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 10. / math.sqrt(self.W.size(1))
        # self.W.data.uniform_(-stdv, stdv)
        self.W.requires_grad = True
        #self.W = torch.tensor(self.W)
        #self.W = torch

    def __call__(self, A, H, activation_flag, NN_flag):
        # -- GCN propagation rule
        # pass

        # Calculate the update and aggregate step
        ## First dot product
        Int_DotProd = torch.matmul(H, self.W)

        if NN_flag == 0:
            Ht_plus1 = torch.matmul(A, Int_DotProd)
        else:
            Ht_plus1 = Int_DotProd

        # # print(A.shape, H.shape, self.W.shape, Int_DotProd.shape)
        # ## Second dot product
        # try:
        #     Int_DotProd_T = torch.transpose(Int_DotProd, 1,2) 
        #     Ht_plus1 = torch.matmul(Int_DotProd_T, self.W)
        # except:
        #     Ht_plus1 = torch.matmul(Int_DotProd, self.W)

        if activation_flag == 1:
            # Define Activation Function
            activation =  nn.ReLU()
            # Apply Activation function
            H_output = activation(Ht_plus1)
        else:
            H_output = Ht_plus1

        return H_output


class GraphPooling:
    """
        Graph pooling layer
    """
    def __init__(self):
        pass

    def __call__(self, H):
        # -- multi-set pooling operator
        #pass

        m,n,p = H.shape

        PoolingT = []
        for i in range(m):
            aux_Matrix = H[i]
            sum_int = aux_Matrix.sum(0)
            sum_int = sum_int.detach().numpy()#.numpy()
            sum_int = [sum_int]
            PoolingT.append(sum_int)

        PoolingT = torch.tensor(PoolingT, requires_grad = True)
        # PoolingT = torch.transpose(PoolingT, 1,2)

        return PoolingT
    

    


class MyModel(nn.Module):
    """
        Regression  model
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # -- initialize layers
        self.FirstLayer = GCN(5,3)
        self.PoolingLayer = GraphPooling()
        self.DenseLayer = GCN(3,1)
        # self.DenseLayer = nn.Linear(9,1)
        #pass

    def forward(self, A, h0):
        
        identity = IdentityTensor(A)

        # print(A.shape, identity.shape)

        # Calculate A_hat = A + I
        adj_hat = A + identity
        d_t   = d_hat(A)
        A_nrm = A_norm(adj_hat, d_t)

        # print(0, A_nrm.shape, h0.shape)
        
        H_first = self.FirstLayer(A_nrm, h0, 1, 0)
        
        # print(1, H_first.shape)
        
        PoolingValues = self.PoolingLayer(H_first)
        
        # print(2, PoolingValues.shape, A_nrm.shape)

        Output = self.DenseLayer(A_nrm, PoolingValues, 0, 1)
        
        #pass

        return Output




# In[17]:


import tensorflow as tf
import torch.optim as optim
from statistics import mean
import torch.nn.functional as F


model = MyModel()

optimizer = optim.SGD(model.parameters(),
                       lr=1e-3)



adjs_aux = adjs[0:5000]
sigs_aux = sigs[0:5000]
prop_aux = prop[0:5000]


avg_L = []
# -- update parameters
for epoch in range(200):

    model.train()
    

    loss = 0

    for i in range(125):
        # print(i)
        
        pred = model(adjs_aux[i*10:(i+1)*10], sigs_aux[i*10:(i+1)*10])
        pred = torch.tensor(pred)

        pred = torch.tensor(pred, dtype=torch.double)

        targets_values = prop_aux[i*10:(i+1)*10]

        targets_values = torch.tensor(targets_values, dtype=torch.double)
        

        loss += F.mse_loss(torch.flatten(model(adjs_aux[i*10:(i+1)*10], sigs_aux[i*10:(i+1)*10])),targets_values)
    
    optimizer.zero_grad()


    # int_L_values.append(loss.item())  

    
    loss.backward()

    optimizer.step()
        
    
    avg_L.append(loss.item())


import matplotlib.pyplot as plt

plt.plot(range(len(avg_L)), avg_L)
plt.xlabel("epochs")
plt.ylabel("Value of loss function")
plt.show()


# In[22]:


plt.plot(range(len(avg_L)), avg_L)
plt.xlabel("epochs")
plt.ylabel("Value of loss function")
plt.ylim((0,1))
plt.show()


# In[20]:


adjs_test = adjs[5000:]
sigs_test = sigs[5000:]
prop_test = prop[5000:]



def test(adjs_test, sigs_test, prop_test):
    model.eval()
    output = model(adjs_test, sigs_test)
    loss_test = F.mse_loss(torch.flatten(model(adjs_test, sigs_test)),prop_test)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))

    plt.scatter(output.detach().numpy(), prop_test.detach().numpy())


    RealValues = prop_test.detach().numpy()

    plt.plot([min(RealValues), max(RealValues)], [min(RealValues), max(RealValues)], "k--")
    plt.xlim([min(RealValues), max(RealValues)])
    plt.ylim([min(RealValues), max(RealValues)])
    plt.xlabel("Predictions")
    plt.ylabel("Real Values")
    
    plt.show()


test(adjs_test, sigs_test, prop_test)

