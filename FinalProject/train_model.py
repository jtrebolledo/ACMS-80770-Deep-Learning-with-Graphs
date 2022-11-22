from utils import *
import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from VGAE_Model import *


def Training_VGAE(AdjacencyMatrix, FeaturesTensor, n_epochs):
    
    FeaturesTensor_Normalize = torch.nn.functional.normalize(FeaturesTensor)
    dim_outputlayer = FeaturesTensor.shape[1]
    dim_inputlayer  = FeaturesTensor.shape[2]
    
    
    n_graphs = AdjacencyMatrix.shape[0]
    Adj_hat = Get_AdjHat(AdjacencyMatrix)
    Adj_hat_reshape = torch.reshape(Adj_hat,(n_graphs,-1))
    
    
    # -- Initialize the model, loss function, and the optimizer
    model = MyModel(dim_inputlayer, dim_outputlayer)


    # MyLoss = nn.KLDivLoss()
    # MyLoss = nn.MSELoss()
    # MyLoss = nn.BCEWithLogitsLoss()
    MyLoss = nn.BCELoss()



    MyOptimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_epoch = []


    batch_size = 100
    # -- update parameters
    for epoch in range(n_epochs):

        cum_loss = 0

        # -- predict
        MyOptimizer.zero_grad()
        pred, mean, logstd, lat_variable  = model(AdjacencyMatrix, FeaturesTensor_Normalize)
        labels = Adj_hat_reshape



        # # # -- loss
        loss = MyLoss(pred, labels)
        # loss = norm*MyLoss(model(adjs[i*batch_size:(i+1)*batch_size], sigs[i*batch_size:(i+1)*batch_size])[0],labels)

        # kl_divergence = 0.5/ pred.size(0) * (1 + 2*logstd - mean**2 - torch.exp(logstd)**2).sum(1).mean()
        # loss -= kl_divergence

        # -- optimize
        loss.backward()
        MyOptimizer.step()

        cum_loss = loss.item()

        loss_epoch.append(cum_loss)

    #     # -- plot loss
    #     X = np.arange(1, n_epochs+1)
    #     fig,ax = plt.subplots(figsize=(10,5))
    #     ax.plot(X, loss_epoch,'r--', lw=2)
    #     ax.tick_params(axis='both', which='major', labelsize=12)
    #     ax.set_xlabel('epoch', fontsize=15)
    #     ax.set_ylabel('MSE Loss', fontsize=15)
    
    
    return n_epochs, loss_epoch, lat_variable

    
