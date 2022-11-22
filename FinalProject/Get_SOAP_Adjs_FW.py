#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import networkx as nx
from Get_FW_Nodes_Edges import *
from ase.db import connect

from GetFeatures_FW import *
import torch
from torch import nn
from torch import optim

import matplotlib.pyplot as plt

from ase.db import connect
from GetFeatures_FW import *
from VGAE_Model import *
from utils import *


# In[2]:


Name_FW_5Al = []
Name_FW_6Al = []
Name_FW_8Al = []
Name_FW_9Al = []
Name_FW_10Al = []
Name_FW_12Al = []
Name_FW_14Al = []
Name_FW_16Al = []
Name_FW_17Al = []
Name_FW_18Al = []
Name_FW_20Al = []
Name_FW_24Al = []
Name_FW_28Al = []
Name_FW_30Al = []
Name_FW_32Al = []
Name_FW_34Al = []
Name_FW_36Al = []
Name_FW_40Al = []
Name_FW_42Al = []
Name_FW_44Al = []
Name_FW_46Al = []
Name_FW_48Al = []
Name_FW_52Al = []
Name_FW_54Al = []
Name_FW_56Al = []
Name_FW_60Al = []
Name_FW_64Al = []
Name_FW_66Al = []
Name_FW_68Al = []
Name_FW_70Al = []
Name_FW_72Al = []
Name_FW_74Al = []
Name_FW_76Al = []
Name_FW_80Al = []
Name_FW_84Al = []
Name_FW_88Al = []
Name_FW_90Al = []
Name_FW_92Al = []
Name_FW_96Al = []
Name_FW_108Al = []
Name_FW_112Al = []
Name_FW_120Al = []
Name_FW_128Al = []
Name_FW_132Al = []
Name_FW_136Al = []
Name_FW_142Al = []
Name_FW_144Al = []
Name_FW_152Al = []
Name_FW_176Al = []
Name_FW_192Al = []
Name_FW_200Al = []
Name_FW_240Al = []
Name_FW_288Al = []
Name_FW_384Al = []
Name_FW_624Al = []
Name_FW_672Al = []
Name_FW_768Al = []
Name_FW_784Al = []
Name_FW_1440Al = []


# In[3]:


df_Frameworks = connect('Database_FW.db')



for row in df_Frameworks.select():
    atoms = row.toatoms()
    nameFW = row.NameFramework
    


    NumberElements = np.array(atoms.get_chemical_symbols())
    Cartesian_coordinates_all = np.array(atoms.get_positions())
    # NumberElements = np.array(atoms.get_chemical_symbols())

    # Identify the index of Oxygen and Silicon
    Index_Element_Si  = np.where(NumberElements == 'Si')[0].copy()
    Index_Element_O   = np.where(NumberElements == 'O')[0].copy()
    Index_Element_Al  = np.where(NumberElements == 'Al')[0].copy()



    Index_Element_Si_Al = Index_Element_Si.copy()
    if len(Index_Element_Al) > 0: 
        for i in Index_Element_Al:
            Index_Element_Si_Al = np.append(Index_Element_Si_Al, i)          



    Edges_total = Get_Edges(atoms)

    labelNodes = []
    Cartesian_coordinates_Si_Al = []
    for i,T in enumerate(Index_Element_Si_Al):
        Cartesian_coordinates_Si_Al.append(list(Cartesian_coordinates_all[T]))
        labelNodes.append("At." + str(i))

    G = nx.Graph()
    G.add_nodes_from(Index_Element_Si_Al)
    G.add_edges_from(Edges_total)
    
    A = nx.to_numpy_matrix(G) #nx.adjacency_matrix(G)
    
    
    atoms_copy = atoms.copy()
    del atoms_copy[[atom.index for atom in atoms_copy if atom.symbol=='O']] 
    N = len(atoms_copy)
    
    
    soap_values = get_SOAPDescriptors(atoms_copy)

    if N == 5:
        if len(Name_FW_5Al) == 0:
            Name_FW_5Al.append(nameFW)
            Adjs_FW_5Al = A
            Tensor_FW_5Al = soap_values
        elif len(Name_FW_5Al) == 1:
            Name_FW_5Al.append(nameFW)
            Adjs_FW_5Al = np.array([Adjs_FW_5Al,A])
            Tensor_FW_5Al = np.array([Tensor_FW_5Al, soap_values])
        else:
            Name_FW_5Al.append(nameFW)
            Adjs_FW_5Al = np.append(Adjs_FW_5Al, [A], axis = 0)
            Tensor_FW_5Al = np.append(Tensor_FW_5Al, [soap_values], axis = 0)
    if N == 6:
        if len(Name_FW_6Al) == 0:
            Name_FW_6Al.append(nameFW)
            Adjs_FW_6Al = A
            Tensor_FW_6Al = soap_values
        elif len(Name_FW_6Al) == 1:
            Name_FW_6Al.append(nameFW)
            Adjs_FW_6Al = np.array([Adjs_FW_6Al,A])
            Tensor_FW_6Al = np.array([Tensor_FW_6Al, soap_values])
        else:
            Name_FW_6Al.append(nameFW)
            Adjs_FW_6Al = np.append(Adjs_FW_6Al, [A], axis = 0)
            Tensor_FW_6Al = np.append(Tensor_FW_6Al, [soap_values], axis = 0)
    if N == 8:
        if len(Name_FW_8Al) == 0:
            Name_FW_8Al.append(nameFW)
            Adjs_FW_8Al = A
            Tensor_FW_8Al = soap_values
        elif len(Name_FW_8Al) == 1:
            Name_FW_8Al.append(nameFW)
            Adjs_FW_8Al = np.array([Adjs_FW_8Al,A])
            Tensor_FW_8Al = np.array([Tensor_FW_8Al, soap_values])
        else:
            Name_FW_8Al.append(nameFW)
            Adjs_FW_8Al = np.append(Adjs_FW_8Al, [A], axis = 0)
            Tensor_FW_8Al = np.append(Tensor_FW_8Al, [soap_values], axis = 0)
    if N == 9:
        if len(Name_FW_9Al) == 0:
            Name_FW_9Al.append(nameFW)
            Adjs_FW_9Al = A
            Tensor_FW_9Al = soap_values
        elif len(Name_FW_9Al) == 1:
            Name_FW_9Al.append(nameFW)
            Adjs_FW_9Al = np.array([Adjs_FW_9Al,A])
            Tensor_FW_9Al = np.array([Tensor_FW_9Al, soap_values])
        else:
            Name_FW_9Al.append(nameFW)
            Adjs_FW_9Al = np.append(Adjs_FW_9Al, [A], axis = 0)
            Tensor_FW_9Al = np.append(Tensor_FW_9Al, [soap_values], axis = 0)
    if N == 10:
        if len(Name_FW_10Al) == 0:
            Name_FW_10Al.append(nameFW)
            Adjs_FW_10Al = A
            Tensor_FW_10Al = soap_values
        elif len(Name_FW_10Al) == 1:
            Name_FW_10Al.append(nameFW)
            Adjs_FW_10Al = np.array([Adjs_FW_10Al,A])
            Tensor_FW_10Al = np.array([Tensor_FW_10Al, soap_values])
        else:
            Name_FW_10Al.append(nameFW)
            Adjs_FW_10Al = np.append(Adjs_FW_10Al, [A], axis = 0)
            Tensor_FW_10Al = np.append(Tensor_FW_10Al, [soap_values], axis = 0)
    if N == 12:
        if len(Name_FW_12Al) == 0:
            Name_FW_12Al.append(nameFW)
            Adjs_FW_12Al = A
            Tensor_FW_12Al = soap_values
        elif len(Name_FW_12Al) == 1:
            Name_FW_12Al.append(nameFW)
            Adjs_FW_12Al = np.array([Adjs_FW_12Al,A])
            Tensor_FW_12Al = np.array([Tensor_FW_12Al, soap_values])
        else:
            Name_FW_12Al.append(nameFW)
            Adjs_FW_12Al = np.append(Adjs_FW_12Al, [A], axis = 0)
            Tensor_FW_12Al = np.append(Tensor_FW_12Al, [soap_values], axis = 0)
    if N == 14:
        if len(Name_FW_14Al) == 0:
            Name_FW_14Al.append(nameFW)
            Adjs_FW_14Al = A
            Tensor_FW_14Al = soap_values
        elif len(Name_FW_14Al) == 1:
            Name_FW_14Al.append(nameFW)
            Adjs_FW_14Al = np.array([Adjs_FW_14Al,A])
            Tensor_FW_14Al = np.array([Tensor_FW_14Al, soap_values])
        else:
            Name_FW_14Al.append(nameFW)
            Adjs_FW_14Al = np.append(Adjs_FW_14Al, [A], axis = 0)
            Tensor_FW_14Al = np.append(Tensor_FW_14Al, [soap_values], axis = 0)
    if N == 16:
        if len(Name_FW_16Al) == 0:
            Name_FW_16Al.append(nameFW)
            Adjs_FW_16Al = A
            Tensor_FW_16Al = soap_values
        elif len(Name_FW_16Al) == 1:
            Name_FW_16Al.append(nameFW)
            Adjs_FW_16Al = np.array([Adjs_FW_16Al,A])
            Tensor_FW_16Al = np.array([Tensor_FW_16Al, soap_values])
        else:
            Name_FW_16Al.append(nameFW)
            Adjs_FW_16Al = np.append(Adjs_FW_16Al, [A], axis = 0)
            Tensor_FW_16Al = np.append(Tensor_FW_16Al, [soap_values], axis = 0)
    if N == 17:
        if len(Name_FW_17Al) == 0:
            Name_FW_17Al.append(nameFW)
            Adjs_FW_17Al = A
            Tensor_FW_17Al = soap_values
        elif len(Name_FW_17Al) == 1:
            Name_FW_17Al.append(nameFW)
            Adjs_FW_17Al = np.array([Adjs_FW_17Al,A])
            Tensor_FW_17Al = np.array([Tensor_FW_17Al, soap_values])
        else:
            Name_FW_17Al.append(nameFW)
            Adjs_FW_17Al = np.append(Adjs_FW_17Al, [A], axis = 0)
            Tensor_FW_17Al = np.append(Tensor_FW_17Al, [soap_values], axis = 0)
    if N == 18:
        if len(Name_FW_18Al) == 0:
            Name_FW_18Al.append(nameFW)
            Adjs_FW_18Al = A
            Tensor_FW_18Al = soap_values
        elif len(Name_FW_18Al) == 1:
            Name_FW_18Al.append(nameFW)
            Adjs_FW_18Al = np.array([Adjs_FW_18Al,A])
            Tensor_FW_18Al = np.array([Tensor_FW_18Al, soap_values])
        else:
            Name_FW_18Al.append(nameFW)
            Adjs_FW_18Al = np.append(Adjs_FW_18Al, [A], axis = 0)
            Tensor_FW_18Al = np.append(Tensor_FW_18Al, [soap_values], axis = 0)
    if N == 20:
        if len(Name_FW_20Al) == 0:
            Name_FW_20Al.append(nameFW)
            Adjs_FW_20Al = A
            Tensor_FW_20Al = soap_values
        elif len(Name_FW_20Al) == 1:
            Name_FW_20Al.append(nameFW)
            Adjs_FW_20Al = np.array([Adjs_FW_20Al,A])
            Tensor_FW_20Al = np.array([Tensor_FW_20Al, soap_values])
        else:
            Name_FW_20Al.append(nameFW)
            Adjs_FW_20Al = np.append(Adjs_FW_20Al, [A], axis = 0)
            Tensor_FW_20Al = np.append(Tensor_FW_20Al, [soap_values], axis = 0)
    if N == 24:
        if len(Name_FW_24Al) == 0:
            Name_FW_24Al.append(nameFW)
            Adjs_FW_24Al = A
            Tensor_FW_24Al = soap_values
        elif len(Name_FW_24Al) == 1:
            Name_FW_24Al.append(nameFW)
            Adjs_FW_24Al = np.array([Adjs_FW_24Al,A])
            Tensor_FW_24Al = np.array([Tensor_FW_24Al, soap_values])
        else:
            Name_FW_24Al.append(nameFW)
            Adjs_FW_24Al = np.append(Adjs_FW_24Al, [A], axis = 0)
            Tensor_FW_24Al = np.append(Tensor_FW_24Al, [soap_values], axis = 0)
    if N == 28:
        if len(Name_FW_28Al) == 0:
            Name_FW_28Al.append(nameFW)
            Adjs_FW_28Al = A
            Tensor_FW_28Al = soap_values
        elif len(Name_FW_28Al) == 1:
            Name_FW_28Al.append(nameFW)
            Adjs_FW_28Al = np.array([Adjs_FW_28Al,A])
            Tensor_FW_28Al = np.array([Tensor_FW_28Al, soap_values])
        else:
            Name_FW_28Al.append(nameFW)
            Adjs_FW_28Al = np.append(Adjs_FW_28Al, [A], axis = 0)
            Tensor_FW_28Al = np.append(Tensor_FW_28Al, [soap_values], axis = 0)
    if N == 30:
        if len(Name_FW_30Al) == 0:
            Name_FW_30Al.append(nameFW)
            Adjs_FW_30Al = A
            Tensor_FW_30Al = soap_values
        elif len(Name_FW_30Al) == 1:
            Name_FW_30Al.append(nameFW)
            Adjs_FW_30Al = np.array([Adjs_FW_30Al,A])
            Tensor_FW_30Al = np.array([Tensor_FW_30Al, soap_values])
        else:
            Name_FW_30Al.append(nameFW)
            Adjs_FW_30Al = np.append(Adjs_FW_30Al, [A], axis = 0)
            Tensor_FW_30Al = np.append(Tensor_FW_30Al, [soap_values], axis = 0)
    if N == 32:
        if len(Name_FW_32Al) == 0:
            Name_FW_32Al.append(nameFW)
            Adjs_FW_32Al = A
            Tensor_FW_32Al = soap_values
        elif len(Name_FW_32Al) == 1:
            Name_FW_32Al.append(nameFW)
            Adjs_FW_32Al = np.array([Adjs_FW_32Al,A])
            Tensor_FW_32Al = np.array([Tensor_FW_32Al, soap_values])
        else:
            Name_FW_32Al.append(nameFW)
            Adjs_FW_32Al = np.append(Adjs_FW_32Al, [A], axis = 0)
            Tensor_FW_32Al = np.append(Tensor_FW_32Al, [soap_values], axis = 0)
    if N == 34:
        if len(Name_FW_34Al) == 0:
            Name_FW_34Al.append(nameFW)
            Adjs_FW_34Al = A
            Tensor_FW_34Al = soap_values
        elif len(Name_FW_34Al) == 1:
            Name_FW_34Al.append(nameFW)
            Adjs_FW_34Al = np.array([Adjs_FW_34Al,A])
            Tensor_FW_34Al = np.array([Tensor_FW_34Al, soap_values])
        else:
            Name_FW_34Al.append(nameFW)
            Adjs_FW_34Al = np.append(Adjs_FW_34Al, [A], axis = 0)
            Tensor_FW_34Al = np.append(Tensor_FW_34Al, [soap_values], axis = 0)
    if N == 36:
        if len(Name_FW_36Al) == 0:
            Name_FW_36Al.append(nameFW)
            Adjs_FW_36Al = A
            Tensor_FW_36Al = soap_values
        elif len(Name_FW_36Al) == 1:
            Name_FW_36Al.append(nameFW)
            Adjs_FW_36Al = np.array([Adjs_FW_36Al,A])
            Tensor_FW_36Al = np.array([Tensor_FW_36Al, soap_values])
        else:
            Name_FW_36Al.append(nameFW)
            Adjs_FW_36Al = np.append(Adjs_FW_36Al, [A], axis = 0)
            Tensor_FW_36Al = np.append(Tensor_FW_36Al, [soap_values], axis = 0)
    if N == 40:
        if len(Name_FW_40Al) == 0:
            Name_FW_40Al.append(nameFW)
            Adjs_FW_40Al = A
            Tensor_FW_40Al = soap_values
        elif len(Name_FW_40Al) == 1:
            Name_FW_40Al.append(nameFW)
            Adjs_FW_40Al = np.array([Adjs_FW_40Al,A])
            Tensor_FW_40Al = np.array([Tensor_FW_40Al, soap_values])
        else:
            Name_FW_40Al.append(nameFW)
            Adjs_FW_40Al = np.append(Adjs_FW_40Al, [A], axis = 0)
            Tensor_FW_40Al = np.append(Tensor_FW_40Al, [soap_values], axis = 0)
    if N == 42:
        if len(Name_FW_42Al) == 0:
            Name_FW_42Al.append(nameFW)
            Adjs_FW_42Al = A
            Tensor_FW_42Al = soap_values
        elif len(Name_FW_42Al) == 1:
            Name_FW_42Al.append(nameFW)
            Adjs_FW_42Al = np.array([Adjs_FW_42Al,A])
            Tensor_FW_42Al = np.array([Tensor_FW_42Al, soap_values])
        else:
            Name_FW_42Al.append(nameFW)
            Adjs_FW_42Al = np.append(Adjs_FW_42Al, [A], axis = 0)
            Tensor_FW_42Al = np.append(Tensor_FW_42Al, [soap_values], axis = 0)
    if N == 44:
        if len(Name_FW_44Al) == 0:
            Name_FW_44Al.append(nameFW)
            Adjs_FW_44Al = A
            Tensor_FW_44Al = soap_values
        elif len(Name_FW_44Al) == 1:
            Name_FW_44Al.append(nameFW)
            Adjs_FW_44Al = np.array([Adjs_FW_44Al,A])
            Tensor_FW_44Al = np.array([Tensor_FW_44Al, soap_values])
        else:
            Name_FW_44Al.append(nameFW)
            Adjs_FW_44Al = np.append(Adjs_FW_44Al, [A], axis = 0)
            Tensor_FW_44Al = np.append(Tensor_FW_44Al, [soap_values], axis = 0)
    if N == 46:
        if len(Name_FW_46Al) == 0:
            Name_FW_46Al.append(nameFW)
            Adjs_FW_46Al = A
            Tensor_FW_46Al = soap_values
        elif len(Name_FW_46Al) == 1:
            Name_FW_46Al.append(nameFW)
            Adjs_FW_46Al = np.array([Adjs_FW_46Al,A])
            Tensor_FW_46Al = np.array([Tensor_FW_46Al, soap_values])
        else:
            Name_FW_46Al.append(nameFW)
            Adjs_FW_46Al = np.append(Adjs_FW_46Al, [A], axis = 0)
            Tensor_FW_46Al = np.append(Tensor_FW_46Al, [soap_values], axis = 0)
    if N == 48:
        if len(Name_FW_48Al) == 0:
            Name_FW_48Al.append(nameFW)
            Adjs_FW_48Al = A
            Tensor_FW_48Al = soap_values
        elif len(Name_FW_48Al) == 1:
            Name_FW_48Al.append(nameFW)
            Adjs_FW_48Al = np.array([Adjs_FW_48Al,A])
            Tensor_FW_48Al = np.array([Tensor_FW_48Al, soap_values])
        else:
            Name_FW_48Al.append(nameFW)
            Adjs_FW_48Al = np.append(Adjs_FW_48Al, [A], axis = 0)
            Tensor_FW_48Al = np.append(Tensor_FW_48Al, [soap_values], axis = 0)
    if N == 52:
        if len(Name_FW_52Al) == 0:
            Name_FW_52Al.append(nameFW)
            Adjs_FW_52Al = A
            Tensor_FW_52Al = soap_values
        elif len(Name_FW_52Al) == 1:
            Name_FW_52Al.append(nameFW)
            Adjs_FW_52Al = np.array([Adjs_FW_52Al,A])
            Tensor_FW_52Al = np.array([Tensor_FW_52Al, soap_values])
        else:
            Name_FW_52Al.append(nameFW)
            Adjs_FW_52Al = np.append(Adjs_FW_52Al, [A], axis = 0)
            Tensor_FW_52Al = np.append(Tensor_FW_52Al, [soap_values], axis = 0)
    if N == 54:
        if len(Name_FW_54Al) == 0:
            Name_FW_54Al.append(nameFW)
            Adjs_FW_54Al = A
            Tensor_FW_54Al = soap_values
        elif len(Name_FW_54Al) == 1:
            Name_FW_54Al.append(nameFW)
            Adjs_FW_54Al = np.array([Adjs_FW_54Al,A])
            Tensor_FW_54Al = np.array([Tensor_FW_54Al, soap_values])
        else:
            Name_FW_54Al.append(nameFW)
            Adjs_FW_54Al = np.append(Adjs_FW_54Al, [A], axis = 0)
            Tensor_FW_54Al = np.append(Tensor_FW_54Al, [soap_values], axis = 0)
    if N == 56:
        if len(Name_FW_56Al) == 0:
            Name_FW_56Al.append(nameFW)
            Adjs_FW_56Al = A
            Tensor_FW_56Al = soap_values
        elif len(Name_FW_56Al) == 1:
            Name_FW_56Al.append(nameFW)
            Adjs_FW_56Al = np.array([Adjs_FW_56Al,A])
            Tensor_FW_56Al = np.array([Tensor_FW_56Al, soap_values])
        else:
            Name_FW_56Al.append(nameFW)
            Adjs_FW_56Al = np.append(Adjs_FW_56Al, [A], axis = 0)
            Tensor_FW_56Al = np.append(Tensor_FW_56Al, [soap_values], axis = 0)
    if N == 60:
        if len(Name_FW_60Al) == 0:
            Name_FW_60Al.append(nameFW)
            Adjs_FW_60Al = A
            Tensor_FW_60Al = soap_values
        elif len(Name_FW_60Al) == 1:
            Name_FW_60Al.append(nameFW)
            Adjs_FW_60Al = np.array([Adjs_FW_60Al,A])
            Tensor_FW_60Al = np.array([Tensor_FW_60Al, soap_values])
        else:
            Name_FW_60Al.append(nameFW)
            Adjs_FW_60Al = np.append(Adjs_FW_60Al, [A], axis = 0)
            Tensor_FW_60Al = np.append(Tensor_FW_60Al, [soap_values], axis = 0)
    if N == 64:
        if len(Name_FW_64Al) == 0:
            Name_FW_64Al.append(nameFW)
            Adjs_FW_64Al = A
            Tensor_FW_64Al = soap_values
        elif len(Name_FW_64Al) == 1:
            Name_FW_64Al.append(nameFW)
            Adjs_FW_64Al = np.array([Adjs_FW_64Al,A])
            Tensor_FW_64Al = np.array([Tensor_FW_64Al, soap_values])
        else:
            Name_FW_64Al.append(nameFW)
            Adjs_FW_64Al = np.append(Adjs_FW_64Al, [A], axis = 0)
            Tensor_FW_64Al = np.append(Tensor_FW_64Al, [soap_values], axis = 0)
    if N == 66:
        if len(Name_FW_66Al) == 0:
            Name_FW_66Al.append(nameFW)
            Adjs_FW_66Al = A
            Tensor_FW_66Al = soap_values
        elif len(Name_FW_66Al) == 1:
            Name_FW_66Al.append(nameFW)
            Adjs_FW_66Al = np.array([Adjs_FW_66Al,A])
            Tensor_FW_66Al = np.array([Tensor_FW_66Al, soap_values])
        else:
            Name_FW_66Al.append(nameFW)
            Adjs_FW_66Al = np.append(Adjs_FW_66Al, [A], axis = 0)
            Tensor_FW_66Al = np.append(Tensor_FW_66Al, [soap_values], axis = 0)
    if N == 68:
        if len(Name_FW_68Al) == 0:
            Name_FW_68Al.append(nameFW)
            Adjs_FW_68Al = A
            Tensor_FW_68Al = soap_values
        elif len(Name_FW_68Al) == 1:
            Name_FW_68Al.append(nameFW)
            Adjs_FW_68Al = np.array([Adjs_FW_68Al,A])
            Tensor_FW_68Al = np.array([Tensor_FW_68Al, soap_values])
        else:
            Name_FW_68Al.append(nameFW)
            Adjs_FW_68Al = np.append(Adjs_FW_68Al, [A], axis = 0)
            Tensor_FW_68Al = np.append(Tensor_FW_68Al, [soap_values], axis = 0)
    if N == 70:
        if len(Name_FW_70Al) == 0:
            Name_FW_70Al.append(nameFW)
            Adjs_FW_70Al = A
            Tensor_FW_70Al = soap_values
        elif len(Name_FW_70Al) == 1:
            Name_FW_70Al.append(nameFW)
            Adjs_FW_70Al = np.array([Adjs_FW_70Al,A])
            Tensor_FW_70Al = np.array([Tensor_FW_70Al, soap_values])
        else:
            Name_FW_70Al.append(nameFW)
            Adjs_FW_70Al = np.append(Adjs_FW_70Al, [A], axis = 0)
            Tensor_FW_70Al = np.append(Tensor_FW_70Al, [soap_values], axis = 0)
    if N == 72:
        if len(Name_FW_72Al) == 0:
            Name_FW_72Al.append(nameFW)
            Adjs_FW_72Al = A
            Tensor_FW_72Al = soap_values
        elif len(Name_FW_72Al) == 1:
            Name_FW_72Al.append(nameFW)
            Adjs_FW_72Al = np.array([Adjs_FW_72Al,A])
            Tensor_FW_72Al = np.array([Tensor_FW_72Al, soap_values])
        else:
            Name_FW_72Al.append(nameFW)
            Adjs_FW_72Al = np.append(Adjs_FW_72Al, [A], axis = 0)
            Tensor_FW_72Al = np.append(Tensor_FW_72Al, [soap_values], axis = 0)
    if N == 74:
        if len(Name_FW_74Al) == 0:
            Name_FW_74Al.append(nameFW)
            Adjs_FW_74Al = A
            Tensor_FW_74Al = soap_values
        elif len(Name_FW_74Al) == 1:
            Name_FW_74Al.append(nameFW)
            Adjs_FW_74Al = np.array([Adjs_FW_74Al,A])
            Tensor_FW_74Al = np.array([Tensor_FW_74Al, soap_values])
        else:
            Name_FW_74Al.append(nameFW)
            Adjs_FW_74Al = np.append(Adjs_FW_74Al, [A], axis = 0)
            Tensor_FW_74Al = np.append(Tensor_FW_74Al, [soap_values], axis = 0)
    if N == 76:
        if len(Name_FW_76Al) == 0:
            Name_FW_76Al.append(nameFW)
            Adjs_FW_76Al = A
            Tensor_FW_76Al = soap_values
        elif len(Name_FW_76Al) == 1:
            Name_FW_76Al.append(nameFW)
            Adjs_FW_76Al = np.array([Adjs_FW_76Al,A])
            Tensor_FW_76Al = np.array([Tensor_FW_76Al, soap_values])
        else:
            Name_FW_76Al.append(nameFW)
            Adjs_FW_76Al = np.append(Adjs_FW_76Al, [A], axis = 0)
            Tensor_FW_76Al = np.append(Tensor_FW_76Al, [soap_values], axis = 0)
    if N == 80:
        if len(Name_FW_80Al) == 0:
            Name_FW_80Al.append(nameFW)
            Adjs_FW_80Al = A
            Tensor_FW_80Al = soap_values
        elif len(Name_FW_80Al) == 1:
            Name_FW_80Al.append(nameFW)
            Adjs_FW_80Al = np.array([Adjs_FW_80Al,A])
            Tensor_FW_80Al = np.array([Tensor_FW_80Al, soap_values])
        else:
            Name_FW_80Al.append(nameFW)
            Adjs_FW_80Al = np.append(Adjs_FW_80Al, [A], axis = 0)
            Tensor_FW_80Al = np.append(Tensor_FW_80Al, [soap_values], axis = 0)
    if N == 84:
        if len(Name_FW_84Al) == 0:
            Name_FW_84Al.append(nameFW)
            Adjs_FW_84Al = A
            Tensor_FW_84Al = soap_values
        elif len(Name_FW_84Al) == 1:
            Name_FW_84Al.append(nameFW)
            Adjs_FW_84Al = np.array([Adjs_FW_84Al,A])
            Tensor_FW_84Al = np.array([Tensor_FW_84Al, soap_values])
        else:
            Name_FW_84Al.append(nameFW)
            Adjs_FW_84Al = np.append(Adjs_FW_84Al, [A], axis = 0)
            Tensor_FW_84Al = np.append(Tensor_FW_84Al, [soap_values], axis = 0)
    if N == 88:
        if len(Name_FW_88Al) == 0:
            Name_FW_88Al.append(nameFW)
            Adjs_FW_88Al = A
            Tensor_FW_88Al = soap_values
        elif len(Name_FW_88Al) == 1:
            Name_FW_88Al.append(nameFW)
            Adjs_FW_88Al = np.array([Adjs_FW_88Al,A])
            Tensor_FW_88Al = np.array([Tensor_FW_88Al, soap_values])
        else:
            Name_FW_88Al.append(nameFW)
            Adjs_FW_88Al = np.append(Adjs_FW_88Al, [A], axis = 0)
            Tensor_FW_88Al = np.append(Tensor_FW_88Al, [soap_values], axis = 0)
    if N == 90:
        if len(Name_FW_90Al) == 0:
            Name_FW_90Al.append(nameFW)
            Adjs_FW_90Al = A
            Tensor_FW_90Al = soap_values
        elif len(Name_FW_90Al) == 1:
            Name_FW_90Al.append(nameFW)
            Adjs_FW_90Al = np.array([Adjs_FW_90Al,A])
            Tensor_FW_90Al = np.array([Tensor_FW_90Al, soap_values])
        else:
            Name_FW_90Al.append(nameFW)
            Adjs_FW_90Al = np.append(Adjs_FW_90Al, [A], axis = 0)
            Tensor_FW_90Al = np.append(Tensor_FW_90Al, [soap_values], axis = 0)
    if N == 92:
        if len(Name_FW_92Al) == 0:
            Name_FW_92Al.append(nameFW)
            Adjs_FW_92Al = A
            Tensor_FW_92Al = soap_values
        elif len(Name_FW_92Al) == 1:
            Name_FW_92Al.append(nameFW)
            Adjs_FW_92Al = np.array([Adjs_FW_92Al,A])
            Tensor_FW_92Al = np.array([Tensor_FW_92Al, soap_values])
        else:
            Name_FW_92Al.append(nameFW)
            Adjs_FW_92Al = np.append(Adjs_FW_92Al, [A], axis = 0)
            Tensor_FW_92Al = np.append(Tensor_FW_92Al, [soap_values], axis = 0)
    if N == 96:
        if len(Name_FW_96Al) == 0:
            Name_FW_96Al.append(nameFW)
            Adjs_FW_96Al = A
            Tensor_FW_96Al = soap_values
        elif len(Name_FW_96Al) == 1:
            Name_FW_96Al.append(nameFW)
            Adjs_FW_96Al = np.array([Adjs_FW_96Al,A])
            Tensor_FW_96Al = np.array([Tensor_FW_96Al, soap_values])
        else:
            Name_FW_96Al.append(nameFW)
            Adjs_FW_96Al = np.append(Adjs_FW_96Al, [A], axis = 0)
            Tensor_FW_96Al = np.append(Tensor_FW_96Al, [soap_values], axis = 0)
    if N == 108:
        if len(Name_FW_108Al) == 0:
            Name_FW_108Al.append(nameFW)
            Adjs_FW_108Al = A
            Tensor_FW_108Al = soap_values
        elif len(Name_FW_108Al) == 1:
            Name_FW_108Al.append(nameFW)
            Adjs_FW_108Al = np.array([Adjs_FW_108Al,A])
            Tensor_FW_108Al = np.array([Tensor_FW_108Al, soap_values])
        else:
            Name_FW_108Al.append(nameFW)
            Adjs_FW_108Al = np.append(Adjs_FW_108Al, [A], axis = 0)
            Tensor_FW_108Al = np.append(Tensor_FW_108Al, [soap_values], axis = 0)
    if N == 112:
        if len(Name_FW_112Al) == 0:
            Name_FW_112Al.append(nameFW)
            Adjs_FW_112Al = A
            Tensor_FW_112Al = soap_values
        elif len(Name_FW_112Al) == 1:
            Name_FW_112Al.append(nameFW)
            Adjs_FW_112Al = np.array([Adjs_FW_112Al,A])
            Tensor_FW_112Al = np.array([Tensor_FW_112Al, soap_values])
        else:
            Name_FW_112Al.append(nameFW)
            Adjs_FW_112Al = np.append(Adjs_FW_112Al, [A], axis = 0)
            Tensor_FW_112Al = np.append(Tensor_FW_112Al, [soap_values], axis = 0)
    if N == 120:
        if len(Name_FW_120Al) == 0:
            Name_FW_120Al.append(nameFW)
            Adjs_FW_120Al = A
            Tensor_FW_120Al = soap_values
        elif len(Name_FW_120Al) == 1:
            Name_FW_120Al.append(nameFW)
            Adjs_FW_120Al = np.array([Adjs_FW_120Al,A])
            Tensor_FW_120Al = np.array([Tensor_FW_120Al, soap_values])
        else:
            Name_FW_120Al.append(nameFW)
            Adjs_FW_120Al = np.append(Adjs_FW_120Al, [A], axis = 0)
            Tensor_FW_120Al = np.append(Tensor_FW_120Al, [soap_values], axis = 0)
    if N == 128:
        if len(Name_FW_128Al) == 0:
            Name_FW_128Al.append(nameFW)
            Adjs_FW_128Al = A
            Tensor_FW_128Al = soap_values
        elif len(Name_FW_128Al) == 1:
            Name_FW_128Al.append(nameFW)
            Adjs_FW_128Al = np.array([Adjs_FW_128Al,A])
            Tensor_FW_128Al = np.array([Tensor_FW_128Al, soap_values])
        else:
            Name_FW_128Al.append(nameFW)
            Adjs_FW_128Al = np.append(Adjs_FW_128Al, [A], axis = 0)
            Tensor_FW_128Al = np.append(Tensor_FW_128Al, [soap_values], axis = 0)
    if N == 132:
        if len(Name_FW_132Al) == 0:
            Name_FW_132Al.append(nameFW)
            Adjs_FW_132Al = A
            Tensor_FW_132Al = soap_values
        elif len(Name_FW_132Al) == 1:
            Name_FW_132Al.append(nameFW)
            Adjs_FW_132Al = np.array([Adjs_FW_132Al,A])
            Tensor_FW_132Al = np.array([Tensor_FW_132Al, soap_values])
        else:
            Name_FW_132Al.append(nameFW)
            Adjs_FW_132Al = np.append(Adjs_FW_132Al, [A], axis = 0)
            Tensor_FW_132Al = np.append(Tensor_FW_132Al, [soap_values], axis = 0)
    if N == 136:
        if len(Name_FW_136Al) == 0:
            Name_FW_136Al.append(nameFW)
            Adjs_FW_136Al = A
            Tensor_FW_136Al = soap_values
        elif len(Name_FW_136Al) == 1:
            Name_FW_136Al.append(nameFW)
            Adjs_FW_136Al = np.array([Adjs_FW_136Al,A])
            Tensor_FW_136Al = np.array([Tensor_FW_136Al, soap_values])
        else:
            Name_FW_136Al.append(nameFW)
            Adjs_FW_136Al = np.append(Adjs_FW_136Al, [A], axis = 0)
            Tensor_FW_136Al = np.append(Tensor_FW_136Al, [soap_values], axis = 0)
    if N == 142:
        if len(Name_FW_142Al) == 0:
            Name_FW_142Al.append(nameFW)
            Adjs_FW_142Al = A
            Tensor_FW_142Al = soap_values
        elif len(Name_FW_142Al) == 1:
            Name_FW_142Al.append(nameFW)
            Adjs_FW_142Al = np.array([Adjs_FW_142Al,A])
            Tensor_FW_142Al = np.array([Tensor_FW_142Al, soap_values])
        else:
            Name_FW_142Al.append(nameFW)
            Adjs_FW_142Al = np.append(Adjs_FW_142Al, [A], axis = 0)
            Tensor_FW_142Al = np.append(Tensor_FW_142Al, [soap_values], axis = 0)
    if N == 144:
        if len(Name_FW_144Al) == 0:
            Name_FW_144Al.append(nameFW)
            Adjs_FW_144Al = A
            Tensor_FW_144Al = soap_values
        elif len(Name_FW_144Al) == 1:
            Name_FW_144Al.append(nameFW)
            Adjs_FW_144Al = np.array([Adjs_FW_144Al,A])
            Tensor_FW_144Al = np.array([Tensor_FW_144Al, soap_values])
        else:
            Name_FW_144Al.append(nameFW)
            Adjs_FW_144Al = np.append(Adjs_FW_144Al, [A], axis = 0)
            Tensor_FW_144Al = np.append(Tensor_FW_144Al, [soap_values], axis = 0)
    if N == 152:
        if len(Name_FW_152Al) == 0:
            Name_FW_152Al.append(nameFW)
            Adjs_FW_152Al = A
            Tensor_FW_152Al = soap_values
        elif len(Name_FW_152Al) == 1:
            Name_FW_152Al.append(nameFW)
            Adjs_FW_152Al = np.array([Adjs_FW_152Al,A])
            Tensor_FW_152Al = np.array([Tensor_FW_152Al, soap_values])
        else:
            Name_FW_152Al.append(nameFW)
            Adjs_FW_152Al = np.append(Adjs_FW_152Al, [A], axis = 0)
            Tensor_FW_152Al = np.append(Tensor_FW_152Al, [soap_values], axis = 0)
    if N == 176:
        if len(Name_FW_176Al) == 0:
            Name_FW_176Al.append(nameFW)
            Adjs_FW_176Al = A
            Tensor_FW_176Al = soap_values
        elif len(Name_FW_176Al) == 1:
            Name_FW_176Al.append(nameFW)
            Adjs_FW_176Al = np.array([Adjs_FW_176Al,A])
            Tensor_FW_176Al = np.array([Tensor_FW_176Al, soap_values])
        else:
            Name_FW_176Al.append(nameFW)
            Adjs_FW_176Al = np.append(Adjs_FW_176Al, [A], axis = 0)
            Tensor_FW_176Al = np.append(Tensor_FW_176Al, [soap_values], axis = 0)
    if N == 192:
        if len(Name_FW_192Al) == 0:
            Name_FW_192Al.append(nameFW)
            Adjs_FW_192Al = A
            Tensor_FW_192Al = soap_values
        elif len(Name_FW_192Al) == 1:
            Name_FW_192Al.append(nameFW)
            Adjs_FW_192Al = np.array([Adjs_FW_192Al,A])
            Tensor_FW_192Al = np.array([Tensor_FW_192Al, soap_values])
        else:
            Name_FW_192Al.append(nameFW)
            Adjs_FW_192Al = np.append(Adjs_FW_192Al, [A], axis = 0)
            Tensor_FW_192Al = np.append(Tensor_FW_192Al, [soap_values], axis = 0)
    if N == 200:
        if len(Name_FW_200Al) == 0:
            Name_FW_200Al.append(nameFW)
            Adjs_FW_200Al = A
            Tensor_FW_200Al = soap_values
        elif len(Name_FW_200Al) == 1:
            Name_FW_200Al.append(nameFW)
            Adjs_FW_200Al = np.array([Adjs_FW_200Al,A])
            Tensor_FW_200Al = np.array([Tensor_FW_200Al, soap_values])
        else:
            Name_FW_200Al.append(nameFW)
            Adjs_FW_200Al = np.append(Adjs_FW_200Al, [A], axis = 0)
            Tensor_FW_200Al = np.append(Tensor_FW_200Al, [soap_values], axis = 0)
    if N == 240:
        if len(Name_FW_240Al) == 0:
            Name_FW_240Al.append(nameFW)
            Adjs_FW_240Al = A
            Tensor_FW_240Al = soap_values
        elif len(Name_FW_240Al) == 1:
            Name_FW_240Al.append(nameFW)
            Adjs_FW_240Al = np.array([Adjs_FW_240Al,A])
            Tensor_FW_240Al = np.array([Tensor_FW_240Al, soap_values])
        else:
            Name_FW_240Al.append(nameFW)
            Adjs_FW_240Al = np.append(Adjs_FW_240Al, [A], axis = 0)
            Tensor_FW_240Al = np.append(Tensor_FW_240Al, [soap_values], axis = 0)
    if N == 288:
        if len(Name_FW_288Al) == 0:
            Name_FW_288Al.append(nameFW)
            Adjs_FW_288Al = A
            Tensor_FW_288Al = soap_values
        elif len(Name_FW_288Al) == 1:
            Name_FW_288Al.append(nameFW)
            Adjs_FW_288Al = np.array([Adjs_FW_288Al,A])
            Tensor_FW_288Al = np.array([Tensor_FW_288Al, soap_values])
        else:
            Name_FW_288Al.append(nameFW)
            Adjs_FW_288Al = np.append(Adjs_FW_288Al, [A], axis = 0)
            Tensor_FW_288Al = np.append(Tensor_FW_288Al, [soap_values], axis = 0)
    if N == 384:
        if len(Name_FW_384Al) == 0:
            Name_FW_384Al.append(nameFW)
            Adjs_FW_384Al = A
            Tensor_FW_384Al = soap_values
        elif len(Name_FW_384Al) == 1:
            Name_FW_384Al.append(nameFW)
            Adjs_FW_384Al = np.array([Adjs_FW_384Al,A])
            Tensor_FW_384Al = np.array([Tensor_FW_384Al, soap_values])
        else:
            Name_FW_384Al.append(nameFW)
            Adjs_FW_384Al = np.append(Adjs_FW_384Al, [A], axis = 0)
            Tensor_FW_384Al = np.append(Tensor_FW_384Al, [soap_values], axis = 0)
    if N == 624:
        if len(Name_FW_624Al) == 0:
            Name_FW_624Al.append(nameFW)
            Adjs_FW_624Al = A
            Tensor_FW_624Al = soap_values
        elif len(Name_FW_624Al) == 1:
            Name_FW_624Al.append(nameFW)
            Adjs_FW_624Al = np.array([Adjs_FW_624Al,A])
            Tensor_FW_624Al = np.array([Tensor_FW_624Al, soap_values])
        else:
            Name_FW_624Al.append(nameFW)
            Adjs_FW_624Al = np.append(Adjs_FW_624Al, [A], axis = 0)
            Tensor_FW_624Al = np.append(Tensor_FW_624Al, [soap_values], axis = 0)
    if N == 672:
        if len(Name_FW_672Al) == 0:
            Name_FW_672Al.append(nameFW)
            Adjs_FW_672Al = A
            Tensor_FW_672Al = soap_values
        elif len(Name_FW_672Al) == 1:
            Name_FW_672Al.append(nameFW)
            Adjs_FW_672Al = np.array([Adjs_FW_672Al,A])
            Tensor_FW_672Al = np.array([Tensor_FW_672Al, soap_values])
        else:
            Name_FW_672Al.append(nameFW)
            Adjs_FW_672Al = np.append(Adjs_FW_672Al, [A], axis = 0)
            Tensor_FW_672Al = np.append(Tensor_FW_672Al, [soap_values], axis = 0)
    if N == 768:
        if len(Name_FW_768Al) == 0:
            Name_FW_768Al.append(nameFW)
            Adjs_FW_768Al = A
            Tensor_FW_768Al = soap_values
        elif len(Name_FW_768Al) == 1:
            Name_FW_768Al.append(nameFW)
            Adjs_FW_768Al = np.array([Adjs_FW_768Al,A])
            Tensor_FW_768Al = np.array([Tensor_FW_768Al, soap_values])
        else:
            Name_FW_768Al.append(nameFW)
            Adjs_FW_768Al = np.append(Adjs_FW_768Al, [A], axis = 0)
            Tensor_FW_768Al = np.append(Tensor_FW_768Al, [soap_values], axis = 0)
    if N == 784:
        if len(Name_FW_784Al) == 0:
            Name_FW_784Al.append(nameFW)
            Adjs_FW_784Al = A
            Tensor_FW_784Al = soap_values
        elif len(Name_FW_784Al) == 1:
            Name_FW_784Al.append(nameFW)
            Adjs_FW_784Al = np.array([Adjs_FW_784Al,A])
            Tensor_FW_784Al = np.array([Tensor_FW_784Al, soap_values])
        else:
            Name_FW_784Al.append(nameFW)
            Adjs_FW_784Al = np.append(Adjs_FW_784Al, [A], axis = 0)
            Tensor_FW_784Al = np.append(Tensor_FW_784Al, [soap_values], axis = 0)
    if N == 1440:
        if len(Name_FW_1440Al) == 0:
            Name_FW_1440Al.append(nameFW)
            Adjs_FW_1440Al = A
            Tensor_FW_1440Al = soap_values
        elif len(Name_FW_1440Al) == 1:
            Name_FW_1440Al.append(nameFW)
            Adjs_FW_1440Al = np.array([Adjs_FW_1440Al,A])
            Tensor_FW_1440Al = np.array([Tensor_FW_1440Al, soap_values])
        else:
            Name_FW_1440Al.append(nameFW)
            Adjs_FW_1440Al = np.append(Adjs_FW_1440Al, [A], axis = 0)
            Tensor_FW_1440Al = np.append(Tensor_FW_1440Al, [soap_values], axis = 0)
            

            
    
    print(nameFW, A.shape, N, soap_values.shape)


# In[4]:


Tensor_FW_5Al = torch.tensor(Tensor_FW_5Al, dtype = torch.float32)
Tensor_FW_6Al = torch.tensor(Tensor_FW_6Al, dtype = torch.float32)
Tensor_FW_8Al = torch.tensor(Tensor_FW_8Al, dtype = torch.float32)
Tensor_FW_9Al = torch.tensor(Tensor_FW_9Al, dtype = torch.float32)
Tensor_FW_10Al = torch.tensor(Tensor_FW_10Al, dtype = torch.float32)
Tensor_FW_12Al = torch.tensor(Tensor_FW_12Al, dtype = torch.float32)
Tensor_FW_14Al = torch.tensor(Tensor_FW_14Al, dtype = torch.float32)
Tensor_FW_16Al = torch.tensor(Tensor_FW_16Al, dtype = torch.float32)
Tensor_FW_17Al = torch.tensor(Tensor_FW_17Al, dtype = torch.float32)
Tensor_FW_18Al = torch.tensor(Tensor_FW_18Al, dtype = torch.float32)
Tensor_FW_20Al = torch.tensor(Tensor_FW_20Al, dtype = torch.float32)
Tensor_FW_24Al = torch.tensor(Tensor_FW_24Al, dtype = torch.float32)
Tensor_FW_28Al = torch.tensor(Tensor_FW_28Al, dtype = torch.float32)
Tensor_FW_30Al = torch.tensor(Tensor_FW_30Al, dtype = torch.float32)
Tensor_FW_32Al = torch.tensor(Tensor_FW_32Al, dtype = torch.float32)
Tensor_FW_34Al = torch.tensor(Tensor_FW_34Al, dtype = torch.float32)
Tensor_FW_36Al = torch.tensor(Tensor_FW_36Al, dtype = torch.float32)
Tensor_FW_40Al = torch.tensor(Tensor_FW_40Al, dtype = torch.float32)
Tensor_FW_42Al = torch.tensor(Tensor_FW_42Al, dtype = torch.float32)
Tensor_FW_44Al = torch.tensor(Tensor_FW_44Al, dtype = torch.float32)
Tensor_FW_46Al = torch.tensor(Tensor_FW_46Al, dtype = torch.float32)
Tensor_FW_48Al = torch.tensor(Tensor_FW_48Al, dtype = torch.float32)
Tensor_FW_52Al = torch.tensor(Tensor_FW_52Al, dtype = torch.float32)
Tensor_FW_54Al = torch.tensor(Tensor_FW_54Al, dtype = torch.float32)
Tensor_FW_56Al = torch.tensor(Tensor_FW_56Al, dtype = torch.float32)
Tensor_FW_60Al = torch.tensor(Tensor_FW_60Al, dtype = torch.float32)
Tensor_FW_64Al = torch.tensor(Tensor_FW_64Al, dtype = torch.float32)
Tensor_FW_66Al = torch.tensor(Tensor_FW_66Al, dtype = torch.float32)
Tensor_FW_68Al = torch.tensor(Tensor_FW_68Al, dtype = torch.float32)
Tensor_FW_70Al = torch.tensor(Tensor_FW_70Al, dtype = torch.float32)
Tensor_FW_72Al = torch.tensor(Tensor_FW_72Al, dtype = torch.float32)
Tensor_FW_74Al = torch.tensor(Tensor_FW_74Al, dtype = torch.float32)
Tensor_FW_76Al = torch.tensor(Tensor_FW_76Al, dtype = torch.float32)
Tensor_FW_80Al = torch.tensor(Tensor_FW_80Al, dtype = torch.float32)
Tensor_FW_84Al = torch.tensor(Tensor_FW_84Al, dtype = torch.float32)
Tensor_FW_88Al = torch.tensor(Tensor_FW_88Al, dtype = torch.float32)
Tensor_FW_90Al = torch.tensor(Tensor_FW_90Al, dtype = torch.float32)
Tensor_FW_92Al = torch.tensor(Tensor_FW_92Al, dtype = torch.float32)
Tensor_FW_96Al = torch.tensor(Tensor_FW_96Al, dtype = torch.float32)
Tensor_FW_108Al = torch.tensor(Tensor_FW_108Al, dtype = torch.float32)
Tensor_FW_112Al = torch.tensor(Tensor_FW_112Al, dtype = torch.float32)
Tensor_FW_120Al = torch.tensor(Tensor_FW_120Al, dtype = torch.float32)
Tensor_FW_128Al = torch.tensor(Tensor_FW_128Al, dtype = torch.float32)
Tensor_FW_132Al = torch.tensor(Tensor_FW_132Al, dtype = torch.float32)
Tensor_FW_136Al = torch.tensor(Tensor_FW_136Al, dtype = torch.float32)
Tensor_FW_142Al = torch.tensor(Tensor_FW_142Al, dtype = torch.float32)
Tensor_FW_144Al = torch.tensor(Tensor_FW_144Al, dtype = torch.float32)
Tensor_FW_152Al = torch.tensor(Tensor_FW_152Al, dtype = torch.float32)
Tensor_FW_176Al = torch.tensor(Tensor_FW_176Al, dtype = torch.float32)
Tensor_FW_192Al = torch.tensor(Tensor_FW_192Al, dtype = torch.float32)
Tensor_FW_200Al = torch.tensor(Tensor_FW_200Al, dtype = torch.float32)
Tensor_FW_240Al = torch.tensor(Tensor_FW_240Al, dtype = torch.float32)
Tensor_FW_288Al = torch.tensor(Tensor_FW_288Al, dtype = torch.float32)
Tensor_FW_384Al = torch.tensor(Tensor_FW_384Al, dtype = torch.float32)
Tensor_FW_624Al = torch.tensor(Tensor_FW_624Al, dtype = torch.float32)
Tensor_FW_672Al = torch.tensor(Tensor_FW_672Al, dtype = torch.float32)
Tensor_FW_768Al = torch.tensor(Tensor_FW_768Al, dtype = torch.float32)
Tensor_FW_784Al = torch.tensor(Tensor_FW_784Al, dtype = torch.float32)
Tensor_FW_1440Al = torch.tensor(Tensor_FW_1440Al, dtype = torch.float32)


# In[5]:


Adjs_FW_5Al = torch.tensor(Adjs_FW_5Al, dtype = torch.float32)
Adjs_FW_6Al = torch.tensor(Adjs_FW_6Al, dtype = torch.float32)
Adjs_FW_8Al = torch.tensor(Adjs_FW_8Al, dtype = torch.float32)
Adjs_FW_9Al = torch.tensor(Adjs_FW_9Al, dtype = torch.float32)
Adjs_FW_10Al = torch.tensor(Adjs_FW_10Al, dtype = torch.float32)
Adjs_FW_12Al = torch.tensor(Adjs_FW_12Al, dtype = torch.float32)
Adjs_FW_14Al = torch.tensor(Adjs_FW_14Al, dtype = torch.float32)
Adjs_FW_16Al = torch.tensor(Adjs_FW_16Al, dtype = torch.float32)
Adjs_FW_17Al = torch.tensor(Adjs_FW_17Al, dtype = torch.float32)
Adjs_FW_18Al = torch.tensor(Adjs_FW_18Al, dtype = torch.float32)
Adjs_FW_20Al = torch.tensor(Adjs_FW_20Al, dtype = torch.float32)
Adjs_FW_24Al = torch.tensor(Adjs_FW_24Al, dtype = torch.float32)
Adjs_FW_28Al = torch.tensor(Adjs_FW_28Al, dtype = torch.float32)
Adjs_FW_30Al = torch.tensor(Adjs_FW_30Al, dtype = torch.float32)
Adjs_FW_32Al = torch.tensor(Adjs_FW_32Al, dtype = torch.float32)
Adjs_FW_34Al = torch.tensor(Adjs_FW_34Al, dtype = torch.float32)
Adjs_FW_36Al = torch.tensor(Adjs_FW_36Al, dtype = torch.float32)
Adjs_FW_40Al = torch.tensor(Adjs_FW_40Al, dtype = torch.float32)
Adjs_FW_42Al = torch.tensor(Adjs_FW_42Al, dtype = torch.float32)
Adjs_FW_44Al = torch.tensor(Adjs_FW_44Al, dtype = torch.float32)
Adjs_FW_46Al = torch.tensor(Adjs_FW_46Al, dtype = torch.float32)
Adjs_FW_48Al = torch.tensor(Adjs_FW_48Al, dtype = torch.float32)
Adjs_FW_52Al = torch.tensor(Adjs_FW_52Al, dtype = torch.float32)
Adjs_FW_54Al = torch.tensor(Adjs_FW_54Al, dtype = torch.float32)
Adjs_FW_56Al = torch.tensor(Adjs_FW_56Al, dtype = torch.float32)
Adjs_FW_60Al = torch.tensor(Adjs_FW_60Al, dtype = torch.float32)
Adjs_FW_64Al = torch.tensor(Adjs_FW_64Al, dtype = torch.float32)
Adjs_FW_66Al = torch.tensor(Adjs_FW_66Al, dtype = torch.float32)
Adjs_FW_68Al = torch.tensor(Adjs_FW_68Al, dtype = torch.float32)
Adjs_FW_70Al = torch.tensor(Adjs_FW_70Al, dtype = torch.float32)
Adjs_FW_72Al = torch.tensor(Adjs_FW_72Al, dtype = torch.float32)
Adjs_FW_74Al = torch.tensor(Adjs_FW_74Al, dtype = torch.float32)
Adjs_FW_76Al = torch.tensor(Adjs_FW_76Al, dtype = torch.float32)
Adjs_FW_80Al = torch.tensor(Adjs_FW_80Al, dtype = torch.float32)
Adjs_FW_84Al = torch.tensor(Adjs_FW_84Al, dtype = torch.float32)
Adjs_FW_88Al = torch.tensor(Adjs_FW_88Al, dtype = torch.float32)
Adjs_FW_90Al = torch.tensor(Adjs_FW_90Al, dtype = torch.float32)
Adjs_FW_92Al = torch.tensor(Adjs_FW_92Al, dtype = torch.float32)
Adjs_FW_96Al = torch.tensor(Adjs_FW_96Al, dtype = torch.float32)
Adjs_FW_108Al = torch.tensor(Adjs_FW_108Al, dtype = torch.float32)
Adjs_FW_112Al = torch.tensor(Adjs_FW_112Al, dtype = torch.float32)
Adjs_FW_120Al = torch.tensor(Adjs_FW_120Al, dtype = torch.float32)
Adjs_FW_128Al = torch.tensor(Adjs_FW_128Al, dtype = torch.float32)
Adjs_FW_132Al = torch.tensor(Adjs_FW_132Al, dtype = torch.float32)
Adjs_FW_136Al = torch.tensor(Adjs_FW_136Al, dtype = torch.float32)
Adjs_FW_142Al = torch.tensor(Adjs_FW_142Al, dtype = torch.float32)
Adjs_FW_144Al = torch.tensor(Adjs_FW_144Al, dtype = torch.float32)
Adjs_FW_152Al = torch.tensor(Adjs_FW_152Al, dtype = torch.float32)
Adjs_FW_176Al = torch.tensor(Adjs_FW_176Al, dtype = torch.float32)
Adjs_FW_192Al = torch.tensor(Adjs_FW_192Al, dtype = torch.float32)
Adjs_FW_200Al = torch.tensor(Adjs_FW_200Al, dtype = torch.float32)
Adjs_FW_240Al = torch.tensor(Adjs_FW_240Al, dtype = torch.float32)
Adjs_FW_288Al = torch.tensor(Adjs_FW_288Al, dtype = torch.float32)
Adjs_FW_384Al = torch.tensor(Adjs_FW_384Al, dtype = torch.float32)
Adjs_FW_624Al = torch.tensor(Adjs_FW_624Al, dtype = torch.float32)
Adjs_FW_672Al = torch.tensor(Adjs_FW_672Al, dtype = torch.float32)
Adjs_FW_768Al = torch.tensor(Adjs_FW_768Al, dtype = torch.float32)
Adjs_FW_784Al = torch.tensor(Adjs_FW_784Al, dtype = torch.float32)
Adjs_FW_1440Al = torch.tensor(Adjs_FW_1440Al, dtype = torch.float32)


# In[6]:


dimension = Tensor_FW_5Al.shape
dimension2 = Adjs_FW_5Al.shape
if len(dimension) == 2:
    Tensor_FW_5Al = torch.reshape(Tensor_FW_5Al, (1,dimension[0],dimension[1]))
    Adjs_FW_5Al   = torch.reshape(Adjs_FW_5Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_6Al.shape
dimension2 = Adjs_FW_6Al.shape
if len(dimension) == 2:
    Tensor_FW_6Al = torch.reshape(Tensor_FW_6Al, (1,dimension[0],dimension[1]))
    Adjs_FW_6Al   = torch.reshape(Adjs_FW_6Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_8Al.shape
dimension2 = Adjs_FW_8Al.shape
if len(dimension) == 2:
    Tensor_FW_8Al = torch.reshape(Tensor_FW_8Al, (1,dimension[0],dimension[1]))
    Adjs_FW_8Al   = torch.reshape(Adjs_FW_8Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_9Al.shape
dimension2 = Adjs_FW_9Al.shape
if len(dimension) == 2:
    Tensor_FW_9Al = torch.reshape(Tensor_FW_9Al, (1,dimension[0],dimension[1]))
    Adjs_FW_9Al   = torch.reshape(Adjs_FW_9Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_10Al.shape
dimension2 = Adjs_FW_10Al.shape
if len(dimension) == 2:
    Tensor_FW_10Al = torch.reshape(Tensor_FW_10Al, (1,dimension[0],dimension[1]))
    Adjs_FW_10Al   = torch.reshape(Adjs_FW_10Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_12Al.shape
dimension2 = Adjs_FW_12Al.shape
if len(dimension) == 2:
    Tensor_FW_12Al = torch.reshape(Tensor_FW_12Al, (1,dimension[0],dimension[1]))
    Adjs_FW_12Al   = torch.reshape(Adjs_FW_12Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_14Al.shape
dimension2 = Adjs_FW_14Al.shape
if len(dimension) == 2:
    Tensor_FW_14Al = torch.reshape(Tensor_FW_14Al, (1,dimension[0],dimension[1]))
    Adjs_FW_14Al   = torch.reshape(Adjs_FW_14Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_16Al.shape
dimension2 = Adjs_FW_16Al.shape
if len(dimension) == 2:
    Tensor_FW_16Al = torch.reshape(Tensor_FW_16Al, (1,dimension[0],dimension[1]))
    Adjs_FW_16Al   = torch.reshape(Adjs_FW_16Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_17Al.shape
dimension2 = Adjs_FW_17Al.shape
if len(dimension) == 2:
    Tensor_FW_17Al = torch.reshape(Tensor_FW_17Al, (1,dimension[0],dimension[1]))
    Adjs_FW_17Al   = torch.reshape(Adjs_FW_17Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_18Al.shape
dimension2 = Adjs_FW_18Al.shape
if len(dimension) == 2:
    Tensor_FW_18Al = torch.reshape(Tensor_FW_18Al, (1,dimension[0],dimension[1]))
    Adjs_FW_18Al   = torch.reshape(Adjs_FW_18Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_20Al.shape
dimension2 = Adjs_FW_20Al.shape
if len(dimension) == 2:
    Tensor_FW_20Al = torch.reshape(Tensor_FW_20Al, (1,dimension[0],dimension[1]))
    Adjs_FW_20Al   = torch.reshape(Adjs_FW_20Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_24Al.shape
dimension2 = Adjs_FW_24Al.shape
if len(dimension) == 2:
    Tensor_FW_24Al = torch.reshape(Tensor_FW_24Al, (1,dimension[0],dimension[1]))
    Adjs_FW_24Al   = torch.reshape(Adjs_FW_24Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_28Al.shape
dimension2 = Adjs_FW_28Al.shape
if len(dimension) == 2:
    Tensor_FW_28Al = torch.reshape(Tensor_FW_28Al, (1,dimension[0],dimension[1]))
    Adjs_FW_28Al   = torch.reshape(Adjs_FW_28Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_30Al.shape
dimension2 = Adjs_FW_30Al.shape
if len(dimension) == 2:
    Tensor_FW_30Al = torch.reshape(Tensor_FW_30Al, (1,dimension[0],dimension[1]))
    Adjs_FW_30Al   = torch.reshape(Adjs_FW_30Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_32Al.shape
dimension2 = Adjs_FW_32Al.shape
if len(dimension) == 2:
    Tensor_FW_32Al = torch.reshape(Tensor_FW_32Al, (1,dimension[0],dimension[1]))
    Adjs_FW_32Al   = torch.reshape(Adjs_FW_32Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_34Al.shape
dimension2 = Adjs_FW_34Al.shape
if len(dimension) == 2:
    Tensor_FW_34Al = torch.reshape(Tensor_FW_34Al, (1,dimension[0],dimension[1]))
    Adjs_FW_34Al   = torch.reshape(Adjs_FW_34Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_36Al.shape
dimension2 = Adjs_FW_36Al.shape
if len(dimension) == 2:
    Tensor_FW_36Al = torch.reshape(Tensor_FW_36Al, (1,dimension[0],dimension[1]))
    Adjs_FW_36Al   = torch.reshape(Adjs_FW_36Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_40Al.shape
dimension2 = Adjs_FW_40Al.shape
if len(dimension) == 2:
    Tensor_FW_40Al = torch.reshape(Tensor_FW_40Al, (1,dimension[0],dimension[1]))
    Adjs_FW_40Al   = torch.reshape(Adjs_FW_40Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_42Al.shape
dimension2 = Adjs_FW_42Al.shape
if len(dimension) == 2:
    Tensor_FW_42Al = torch.reshape(Tensor_FW_42Al, (1,dimension[0],dimension[1]))
    Adjs_FW_42Al   = torch.reshape(Adjs_FW_42Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_44Al.shape
dimension2 = Adjs_FW_44Al.shape
if len(dimension) == 2:
    Tensor_FW_44Al = torch.reshape(Tensor_FW_44Al, (1,dimension[0],dimension[1]))
    Adjs_FW_44Al   = torch.reshape(Adjs_FW_44Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_46Al.shape
dimension2 = Adjs_FW_46Al.shape
if len(dimension) == 2:
    Tensor_FW_46Al = torch.reshape(Tensor_FW_46Al, (1,dimension[0],dimension[1]))
    Adjs_FW_46Al   = torch.reshape(Adjs_FW_46Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_48Al.shape
dimension2 = Adjs_FW_48Al.shape
if len(dimension) == 2:
    Tensor_FW_48Al = torch.reshape(Tensor_FW_48Al, (1,dimension[0],dimension[1]))
    Adjs_FW_48Al   = torch.reshape(Adjs_FW_48Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_52Al.shape
dimension2 = Adjs_FW_52Al.shape
if len(dimension) == 2:
    Tensor_FW_52Al = torch.reshape(Tensor_FW_52Al, (1,dimension[0],dimension[1]))
    Adjs_FW_52Al   = torch.reshape(Adjs_FW_52Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_54Al.shape
dimension2 = Adjs_FW_54Al.shape
if len(dimension) == 2:
    Tensor_FW_54Al = torch.reshape(Tensor_FW_54Al, (1,dimension[0],dimension[1]))
    Adjs_FW_54Al   = torch.reshape(Adjs_FW_54Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_56Al.shape
dimension2 = Adjs_FW_56Al.shape
if len(dimension) == 2:
    Tensor_FW_56Al = torch.reshape(Tensor_FW_56Al, (1,dimension[0],dimension[1]))
    Adjs_FW_56Al   = torch.reshape(Adjs_FW_56Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_60Al.shape
dimension2 = Adjs_FW_60Al.shape
if len(dimension) == 2:
    Tensor_FW_60Al = torch.reshape(Tensor_FW_60Al, (1,dimension[0],dimension[1]))
    Adjs_FW_60Al   = torch.reshape(Adjs_FW_60Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_64Al.shape
dimension2 = Adjs_FW_64Al.shape
if len(dimension) == 2:
    Tensor_FW_64Al = torch.reshape(Tensor_FW_64Al, (1,dimension[0],dimension[1]))
    Adjs_FW_64Al   = torch.reshape(Adjs_FW_64Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_66Al.shape
dimension2 = Adjs_FW_66Al.shape
if len(dimension) == 2:
    Tensor_FW_66Al = torch.reshape(Tensor_FW_66Al, (1,dimension[0],dimension[1]))
    Adjs_FW_66Al   = torch.reshape(Adjs_FW_66Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_68Al.shape
dimension2 = Adjs_FW_68Al.shape
if len(dimension) == 2:
    Tensor_FW_68Al = torch.reshape(Tensor_FW_68Al, (1,dimension[0],dimension[1]))
    Adjs_FW_68Al   = torch.reshape(Adjs_FW_68Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_70Al.shape
dimension2 = Adjs_FW_70Al.shape
if len(dimension) == 2:
    Tensor_FW_70Al = torch.reshape(Tensor_FW_70Al, (1,dimension[0],dimension[1]))
    Adjs_FW_70Al   = torch.reshape(Adjs_FW_70Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_72Al.shape
dimension2 = Adjs_FW_72Al.shape
if len(dimension) == 2:
    Tensor_FW_72Al = torch.reshape(Tensor_FW_72Al, (1,dimension[0],dimension[1]))
    Adjs_FW_72Al   = torch.reshape(Adjs_FW_72Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_74Al.shape
dimension2 = Adjs_FW_74Al.shape
if len(dimension) == 2:
    Tensor_FW_74Al = torch.reshape(Tensor_FW_74Al, (1,dimension[0],dimension[1]))
    Adjs_FW_74Al   = torch.reshape(Adjs_FW_74Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_76Al.shape
dimension2 = Adjs_FW_76Al.shape
if len(dimension) == 2:
    Tensor_FW_76Al = torch.reshape(Tensor_FW_76Al, (1,dimension[0],dimension[1]))
    Adjs_FW_76Al   = torch.reshape(Adjs_FW_76Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_80Al.shape
dimension2 = Adjs_FW_80Al.shape
if len(dimension) == 2:
    Tensor_FW_80Al = torch.reshape(Tensor_FW_80Al, (1,dimension[0],dimension[1]))
    Adjs_FW_80Al   = torch.reshape(Adjs_FW_80Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_84Al.shape
dimension2 = Adjs_FW_84Al.shape
if len(dimension) == 2:
    Tensor_FW_84Al = torch.reshape(Tensor_FW_84Al, (1,dimension[0],dimension[1]))
    Adjs_FW_84Al   = torch.reshape(Adjs_FW_84Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_88Al.shape
dimension2 = Adjs_FW_88Al.shape
if len(dimension) == 2:
    Tensor_FW_88Al = torch.reshape(Tensor_FW_88Al, (1,dimension[0],dimension[1]))
    Adjs_FW_88Al   = torch.reshape(Adjs_FW_88Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_90Al.shape
dimension2 = Adjs_FW_90Al.shape
if len(dimension) == 2:
    Tensor_FW_90Al = torch.reshape(Tensor_FW_90Al, (1,dimension[0],dimension[1]))
    Adjs_FW_90Al   = torch.reshape(Adjs_FW_90Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_92Al.shape
dimension2 = Adjs_FW_92Al.shape
if len(dimension) == 2:
    Tensor_FW_92Al = torch.reshape(Tensor_FW_92Al, (1,dimension[0],dimension[1]))
    Adjs_FW_92Al   = torch.reshape(Adjs_FW_92Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_96Al.shape
dimension2 = Adjs_FW_96Al.shape
if len(dimension) == 2:
    Tensor_FW_96Al = torch.reshape(Tensor_FW_96Al, (1,dimension[0],dimension[1]))
    Adjs_FW_96Al   = torch.reshape(Adjs_FW_96Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_108Al.shape
dimension2 = Adjs_FW_108Al.shape
if len(dimension) == 2:
    Tensor_FW_108Al = torch.reshape(Tensor_FW_108Al, (1,dimension[0],dimension[1]))
    Adjs_FW_108Al   = torch.reshape(Adjs_FW_108Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_112Al.shape
dimension2 = Adjs_FW_112Al.shape
if len(dimension) == 2:
    Tensor_FW_112Al = torch.reshape(Tensor_FW_112Al, (1,dimension[0],dimension[1]))
    Adjs_FW_112Al   = torch.reshape(Adjs_FW_112Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_120Al.shape
dimension2 = Adjs_FW_120Al.shape
if len(dimension) == 2:
    Tensor_FW_120Al = torch.reshape(Tensor_FW_120Al, (1,dimension[0],dimension[1]))
    Adjs_FW_120Al   = torch.reshape(Adjs_FW_120Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_128Al.shape
dimension2 = Adjs_FW_128Al.shape
if len(dimension) == 2:
    Tensor_FW_128Al = torch.reshape(Tensor_FW_128Al, (1,dimension[0],dimension[1]))
    Adjs_FW_128Al   = torch.reshape(Adjs_FW_128Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_132Al.shape
dimension2 = Adjs_FW_132Al.shape
if len(dimension) == 2:
    Tensor_FW_132Al = torch.reshape(Tensor_FW_132Al, (1,dimension[0],dimension[1]))
    Adjs_FW_132Al   = torch.reshape(Adjs_FW_132Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_136Al.shape
dimension2 = Adjs_FW_136Al.shape
if len(dimension) == 2:
    Tensor_FW_136Al = torch.reshape(Tensor_FW_136Al, (1,dimension[0],dimension[1]))
    Adjs_FW_136Al   = torch.reshape(Adjs_FW_136Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_142Al.shape
dimension2 = Adjs_FW_142Al.shape
if len(dimension) == 2:
    Tensor_FW_142Al = torch.reshape(Tensor_FW_142Al, (1,dimension[0],dimension[1]))
    Adjs_FW_142Al   = torch.reshape(Adjs_FW_142Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_144Al.shape
dimension2 = Adjs_FW_144Al.shape
if len(dimension) == 2:
    Tensor_FW_144Al = torch.reshape(Tensor_FW_144Al, (1,dimension[0],dimension[1]))
    Adjs_FW_144Al   = torch.reshape(Adjs_FW_144Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_152Al.shape
dimension2 = Adjs_FW_152Al.shape
if len(dimension) == 2:
    Tensor_FW_152Al = torch.reshape(Tensor_FW_152Al, (1,dimension[0],dimension[1]))
    Adjs_FW_152Al   = torch.reshape(Adjs_FW_152Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_176Al.shape
dimension2 = Adjs_FW_176Al.shape
if len(dimension) == 2:
    Tensor_FW_176Al = torch.reshape(Tensor_FW_176Al, (1,dimension[0],dimension[1]))
    Adjs_FW_176Al   = torch.reshape(Adjs_FW_176Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_192Al.shape
dimension2 = Adjs_FW_192Al.shape
if len(dimension) == 2:
    Tensor_FW_192Al = torch.reshape(Tensor_FW_192Al, (1,dimension[0],dimension[1]))
    Adjs_FW_192Al   = torch.reshape(Adjs_FW_192Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_200Al.shape
dimension2 = Adjs_FW_200Al.shape
if len(dimension) == 2:
    Tensor_FW_200Al = torch.reshape(Tensor_FW_200Al, (1,dimension[0],dimension[1]))
    Adjs_FW_200Al   = torch.reshape(Adjs_FW_200Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_240Al.shape
dimension2 = Adjs_FW_240Al.shape
if len(dimension) == 2:
    Tensor_FW_240Al = torch.reshape(Tensor_FW_240Al, (1,dimension[0],dimension[1]))
    Adjs_FW_240Al   = torch.reshape(Adjs_FW_240Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_288Al.shape
dimension2 = Adjs_FW_288Al.shape
if len(dimension) == 2:
    Tensor_FW_288Al = torch.reshape(Tensor_FW_288Al, (1,dimension[0],dimension[1]))
    Adjs_FW_288Al   = torch.reshape(Adjs_FW_288Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_384Al.shape
dimension2 = Adjs_FW_384Al.shape
if len(dimension) == 2:
    Tensor_FW_384Al = torch.reshape(Tensor_FW_384Al, (1,dimension[0],dimension[1]))
    Adjs_FW_384Al   = torch.reshape(Adjs_FW_384Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_624Al.shape
dimension2 = Adjs_FW_624Al.shape
if len(dimension) == 2:
    Tensor_FW_624Al = torch.reshape(Tensor_FW_624Al, (1,dimension[0],dimension[1]))
    Adjs_FW_624Al   = torch.reshape(Adjs_FW_624Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_672Al.shape
dimension2 = Adjs_FW_672Al.shape
if len(dimension) == 2:
    Tensor_FW_672Al = torch.reshape(Tensor_FW_672Al, (1,dimension[0],dimension[1]))
    Adjs_FW_672Al   = torch.reshape(Adjs_FW_672Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_768Al.shape
dimension2 = Adjs_FW_768Al.shape
if len(dimension) == 2:
    Tensor_FW_768Al = torch.reshape(Tensor_FW_768Al, (1,dimension[0],dimension[1]))
    Adjs_FW_768Al   = torch.reshape(Adjs_FW_768Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_784Al.shape
dimension2 = Adjs_FW_784Al.shape
if len(dimension) == 2:
    Tensor_FW_784Al = torch.reshape(Tensor_FW_784Al, (1,dimension[0],dimension[1]))
    Adjs_FW_784Al   = torch.reshape(Adjs_FW_784Al, (1,dimension2[0],dimension2[1]))
    
dimension = Tensor_FW_1440Al.shape
dimension2 = Adjs_FW_1440Al.shape
if len(dimension) == 2:
    Tensor_FW_1440Al = torch.reshape(Tensor_FW_1440Al, (1,dimension[0],dimension[1]))
    Adjs_FW_1440Al   = torch.reshape(Adjs_FW_1440Al, (1,dimension2[0],dimension2[1]))
    


# In[7]:


print(Tensor_FW_5Al.shape)
print(Tensor_FW_6Al.shape)
print(Tensor_FW_8Al.shape)
print(Tensor_FW_9Al.shape)
print(Tensor_FW_10Al.shape)
print(Tensor_FW_12Al.shape)
print(Tensor_FW_14Al.shape)
print(Tensor_FW_16Al.shape)
print(Tensor_FW_17Al.shape)
print(Tensor_FW_18Al.shape)
print(Tensor_FW_20Al.shape)
print(Tensor_FW_24Al.shape)
print(Tensor_FW_28Al.shape)
print(Tensor_FW_30Al.shape)
print(Tensor_FW_32Al.shape)
print(Tensor_FW_34Al.shape)
print(Tensor_FW_36Al.shape)
print(Tensor_FW_40Al.shape)
print(Tensor_FW_42Al.shape)
print(Tensor_FW_44Al.shape)
print(Tensor_FW_46Al.shape)
print(Tensor_FW_48Al.shape)
print(Tensor_FW_52Al.shape)
print(Tensor_FW_54Al.shape)
print(Tensor_FW_56Al.shape)
print(Tensor_FW_60Al.shape)
print(Tensor_FW_64Al.shape)
print(Tensor_FW_66Al.shape)
print(Tensor_FW_68Al.shape)
print(Tensor_FW_70Al.shape)
print(Tensor_FW_72Al.shape)
print(Tensor_FW_74Al.shape)
print(Tensor_FW_76Al.shape)
print(Tensor_FW_80Al.shape)
print(Tensor_FW_84Al.shape)
print(Tensor_FW_88Al.shape)
print(Tensor_FW_90Al.shape)
print(Tensor_FW_92Al.shape)
print(Tensor_FW_96Al.shape)
print(Tensor_FW_108Al.shape)
print(Tensor_FW_112Al.shape)
print(Tensor_FW_120Al.shape)
print(Tensor_FW_128Al.shape)
print(Tensor_FW_132Al.shape)
print(Tensor_FW_136Al.shape)
print(Tensor_FW_142Al.shape)
print(Tensor_FW_144Al.shape)
print(Tensor_FW_152Al.shape)
print(Tensor_FW_176Al.shape)
print(Tensor_FW_192Al.shape)
print(Tensor_FW_200Al.shape)
print(Tensor_FW_240Al.shape)
print(Tensor_FW_288Al.shape)
print(Tensor_FW_384Al.shape)
print(Tensor_FW_624Al.shape)
print(Tensor_FW_672Al.shape)
print(Tensor_FW_768Al.shape)
print(Tensor_FW_784Al.shape)
print(Tensor_FW_1440Al.shape)


# In[8]:


from train_model import *
from RedDimension import *


# In[9]:


XY_variable_sum_max_avg = []
NameFW_Order            = []

def GetListReduceFeatures(Name_FW_Current, NameFW_Order, XY_variable_sum_max_avg, lat_var_sum, lat_var_max, lat_var_avg):
    
#     print(len(Name_FW_Current), Name_FW_Current)
    
    for i in range(len(Name_FW_Current)):
        NameFW_Order.append(Name_FW_Current[i])

        aux_variable = [Name_FW_Current[i], lat_var_sum[i][0].item(), lat_var_sum[i][1].item(), 
                       lat_var_max[i][0].item(), lat_var_max[i][1].item(),
                       lat_var_avg[i][0].item(), lat_var_avg[i][1].item()]

        XY_variable_sum_max_avg.append(aux_variable)
    


# In[10]:


n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_5Al, Tensor_FW_5Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss = np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_5Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)


n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_6Al, Tensor_FW_6Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_6Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_8Al, Tensor_FW_8Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_8Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_9Al, Tensor_FW_9Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_9Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_10Al, Tensor_FW_10Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_10Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_12Al, Tensor_FW_12Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_12Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_14Al, Tensor_FW_14Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_14Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_16Al, Tensor_FW_16Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_16Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_17Al, Tensor_FW_17Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_17Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_18Al, Tensor_FW_18Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_18Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_20Al, Tensor_FW_20Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_20Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_24Al, Tensor_FW_24Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_24Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_28Al, Tensor_FW_28Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_28Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_30Al, Tensor_FW_30Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_30Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_32Al, Tensor_FW_32Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_32Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_34Al, Tensor_FW_34Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_34Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_36Al, Tensor_FW_36Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_36Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_40Al, Tensor_FW_40Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_40Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_42Al, Tensor_FW_42Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_42Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_44Al, Tensor_FW_44Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_44Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_46Al, Tensor_FW_46Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_46Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_48Al, Tensor_FW_48Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_48Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_52Al, Tensor_FW_52Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_52Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_54Al, Tensor_FW_54Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_54Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_56Al, Tensor_FW_56Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_56Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_60Al, Tensor_FW_60Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_60Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_64Al, Tensor_FW_64Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_64Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_66Al, Tensor_FW_66Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_66Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_68Al, Tensor_FW_68Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_68Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_70Al, Tensor_FW_70Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_70Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_72Al, Tensor_FW_72Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_72Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_74Al, Tensor_FW_74Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_74Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_76Al, Tensor_FW_76Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_76Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_80Al, Tensor_FW_80Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_80Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_84Al, Tensor_FW_84Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_84Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_88Al, Tensor_FW_88Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_88Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_90Al, Tensor_FW_90Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_90Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_92Al, Tensor_FW_92Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_92Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_96Al, Tensor_FW_96Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_96Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_108Al, Tensor_FW_108Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_108Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_112Al, Tensor_FW_112Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_112Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_120Al, Tensor_FW_120Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_120Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_128Al, Tensor_FW_128Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_128Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_132Al, Tensor_FW_132Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_132Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_136Al, Tensor_FW_136Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_136Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_142Al, Tensor_FW_142Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_142Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_144Al, Tensor_FW_144Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_144Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_152Al, Tensor_FW_152Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_152Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_176Al, Tensor_FW_176Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_176Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_192Al, Tensor_FW_192Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_192Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_200Al, Tensor_FW_200Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_200Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_240Al, Tensor_FW_240Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_240Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_288Al, Tensor_FW_288Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_288Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_384Al, Tensor_FW_384Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_384Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_624Al, Tensor_FW_624Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_624Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_672Al, Tensor_FW_672Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_672Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_768Al, Tensor_FW_768Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_768Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_784Al, Tensor_FW_784Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_784Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)
n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_1440Al, Tensor_FW_1440Al, 200)
lat_variable_pooling_sum, lat_variable_pooling_max, lat_variable_pooling_avg = Get_LatenteVariableReduce(lat_variable)

AccumulationLoss += np.array(loss_x_epoch)

GetListReduceFeatures(Name_FW_1440Al, NameFW_Order, XY_variable_sum_max_avg, lat_variable_pooling_sum, 
                      lat_variable_pooling_max, lat_variable_pooling_avg)


# In[23]:


n_epochs, loss_x_epoch, lat_variable = Training_VGAE(Adjs_FW_24Al, Tensor_FW_24Al, 200)

# -- plot loss
X = np.arange(1, n_epochs+1)
fig,ax = plt.subplots(figsize=(10,5))
ax.plot(X, loss_x_epoch,'r--', lw=1.5)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('epoch', fontsize=15)
ax.set_ylabel('BCE Loss', fontsize=15)
plt.title("Loss Function of a set of Zeolite Frameworks", fontsize = 20)
plt.show()


fig,ax = plt.subplots(figsize=(10,5))
ax.plot(X, loss_x_epoch, lw=1.5)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('epoch', fontsize=15)
ax.set_ylabel('BCE Loss', fontsize=15)
plt.title("Loss Function of a set of Zeolite Frameworks", fontsize = 20)
plt.show()


# In[24]:


# -- plot loss
X = np.arange(1, n_epochs+1)
fig,ax = plt.subplots(figsize=(10,5))
ax.plot(X, AccumulationLoss/59,'r--', lw=1.5)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('epoch', fontsize=15)
ax.set_ylabel('BCE Loss', fontsize=15)
plt.title("Accumulative Loss Function", fontsize = 20)
plt.show()


# -- plot loss
X = np.arange(1, n_epochs+1)
fig,ax = plt.subplots(figsize=(10,5))
ax.plot(X, AccumulationLoss/59, lw=1.5)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('epoch', fontsize=15)
ax.set_ylabel('BCE Loss', fontsize=15)
plt.title("Accumulative Loss Function", fontsize = 20)
plt.show()


# In[14]:


import pandas as pd 
cols  = ['Framework', 'Lat_Var_Sum_X', 'Lat_Var_Sum_Y', 
         'Lat_Var_Max_X', 'Lat_Var_Max_Y', 'Lat_Var_Avg_X', 'Lat_Var_Avg_Y']


Framework_df = pd.DataFrame(columns=cols)


for i in range(len(NameFW_Order)):
    
    auxArray = np.array(XY_variable_sum_max_avg[i])
    Framework_df.loc[len(Framework_df.index)] = auxArray


# In[16]:


Framework_df.to_csv("Dataframe_FWs_2DimensionAnalysis.csv")

