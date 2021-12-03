#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:49:05 2021

@author: illusionist
"""

# util.py

import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import scipy.sparse as sp
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
import networkx as nx
from csv import DictReader
from csv import reader
import numpy as np
import pandas as pd
import xlrd
import csv
import os
import csv
import xlrd
import _thread
import time
import torch as th
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

"""
    this function creates a dictionary for entity(drug/protein/species/indication) sent as argument
"""

def create_dictionary(path):
    
    Compound_dict={}
    Disease_dict={}
    Gene_dict={}
    Anatomy_dict={}
    # dictionary's key will content entity object's id(drug id/protein id)
    # value will contain the index id where corresponding entity object's data will be found in adjacency and
    # commuting matrix
    count_Compound=0
    count_Disease=0
    count_Gene=0
    count_Anatomy=0
    with open(path) as f:
        content = f.readlines()
        for x in content:
            x = x.split()
            if 'Disease::' in x[0]:
              Disease_dict[x[0]]=count_Disease
              count_Disease+=1
            elif 'Compound::' in x[0]:
              Compound_dict[x[0]]=count_Compound
              count_Compound+=1
            elif 'Gene::' in x[0]:
              Gene_dict[x[0]]=count_Gene
              count_Gene+=1
            elif 'Anatomy::' in x[0]:
              Anatomy_dict[x[0]]=count_Anatomy
              count_Anatomy+=1

    return Compound_dict, Disease_dict, Gene_dict, Anatomy_dict

def create_label(path, mat, dim1, dim2, rowCount, columnCount):
    
    labels = np.zeros((rowCount,columnCount))
    count=0
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    for i in range(dim1):
      for j in range(dim2):
        if mat[i][j]==1:
          labels[count] = np.array([i, j])
          count+=1   

    print('count',count)
    return labels, count

def create_non_interactions(path, mat, dim1, dim2, rowCount, columnCount):
    
    non_interactions = np.zeros((rowCount,columnCount))
    count=0
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    for i in range(dim1):
      for j in range(dim2):
        if mat[i][j]==0:
          non_interactions[count] = np.array([i, j])
          count+=1   

    print('count',count)
    return non_interactions

def create_adjacency_matrix(path, col1, dict1, col2, dict2, AreSameEntities):
    
    adj_mat=[[]]
    
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows, cols= (max(dict1.values())+1, max(dict2.values())+1)
    adj_mat = [[0] * cols for i in range(rows)]
    
    #dfs = pd.read_excel(path, sheetname=None)
    
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    with open(path) as fd:
      rd = csv.reader(fd, delimiter="\t", quotechar='"')
      for row in rd:
          if col1 in row[0] and col2 in row[2]:
            adj_mat[dict1[row[0]]][dict2[row[2]]]=1
    
    return adj_mat


def load_drug_data(remove_self_loop):
    assert not remove_self_loop
    
    # FT 110621 constructing dictionaries
    Compound_dict, Disease_dict, Gene_dict, Anatomy_dict=create_dictionary('/content/drive/MyDrive/Anti-COVID (1)/data/entities.tsv')
    
    # FT 110621 constructing adjacency matrices
    d_VS_ds = np.load('/content/drive/MyDrive/Anti-COVID (1)/data/labels.npy')
    
    d_VS_ds = np.array(d_VS_ds)

    count=0
    for i in range(9572):
      for j in range(3976):
        if d_VS_ds[i][j]==1:
          count+=1

    print('count',count)
    d_VS_d = create_adjacency_matrix('/content/drive/MyDrive/Anti-COVID (1)/data/drkg.tsv', 'Compound::', Compound_dict, 'Compound::', Compound_dict, True)
    ds_VS_ds = create_adjacency_matrix('/content/drive/MyDrive/Anti-COVID (1)/data/drkg.tsv', 'Disease::', Disease_dict, 'Disease::', Disease_dict, True)

    labels, numOfInteractions = create_label('/content/drive/MyDrive/Anti-COVID (1)/data/drkg.tsv', d_VS_ds, 9572, 3976, count, 2)
    print('labels',labels.shape)
    print('numOfInteractions',numOfInteractions)

    numOfNonInteractions=(9572*3976)-numOfInteractions
    print('numOfNonInteractions',numOfNonInteractions)
    non_interactions = create_non_interactions('/content/drive/MyDrive/Anti-COVID (1)/data/drkg.tsv', d_VS_ds, 9572, 3976, numOfNonInteractions, 2)
    print('non_interactions',non_interactions.shape)

    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    
    train_arr_label, test_arr_label = train_test_split(labels, test_size=0.2, random_state=7)
    #d_VS_ds = np.load('/content/drive/MyDrive/Anti-COVID (1)/data/labels.npy')
    print('train and test array label created')
    print('train_arrr_label',len(train_arr_label))
    print('test_arrr_label',len(test_arr_label))

    number_of_rows = non_interactions.shape[0]
    random_indices = np.random.choice(number_of_rows, size=numOfInteractions, replace=False)
    non_interactions = non_interactions[random_indices, :]
    print('non_interactions',non_interactions)

    train_arr_non_interactions, test_arr_non_interactions = train_test_split(non_interactions, test_size=0.2, random_state=7)
    #d_VS_ds = np.load('/content/drive/MyDrive/Anti-COVID (1)/data/labels.npy')
    print('train and test array non_interactions created')
    print('train_arr_non_interactions',len(train_arr_non_interactions))
    print('test_arr_non_interactions',len(test_arr_non_interactions))
    
    """
    train_arr = train_arr_label+train_arr_non_interactions
    test_arr = test_arr_label+test_arr_non_interactions

    print('train_arr',len(train_arr), train_arr)
    print('test_arr',len(test_arr), test_arr)
    """
    train_arr = []

    for i in range(len(train_arr_label)): 
      train_arr.append(train_arr_label[i]) 

    for i in range(len(train_arr_non_interactions)): 
      train_arr.append(train_arr_non_interactions[i])
    
    test_arr = []

    for i in range(len(test_arr_label)): 
      test_arr.append(test_arr_label[i]) 

    for i in range(len(test_arr_non_interactions)): 
      test_arr.append(test_arr_non_interactions[i])

    print('train and test array created')
    print('train_arr',len(train_arr), type(train_arr))
    print('test_arr',len(test_arr), type(test_arr))

    random.shuffle(train_arr)
    random.shuffle(test_arr)

    print('train and test array created')
    print('train_arr',len(train_arr), type(train_arr))
    print('test_arr',len(test_arr), type(test_arr))

    # FT 110621 constructing dgl heterograph
    hg = dgl.heterograph({
        ('drug', 'dds', 'disease'): d_VS_ds.nonzero(),
        ('drug', 'ddi', 'drug'): d_VS_d.nonzero(),
        ('disease', 'dsds', 'disease'): ds_VS_ds.nonzero()
    })
   
   # FT 110621 conversion to homogeneous, undirected graph
    print('graph created')
    hg = dgl.to_homogeneous(hg)
    hg = hg.to_networkx().to_undirected()
    
    return hg, d_VS_ds, Compound_dict, Disease_dict, train_arr, test_arr
            
    

def load_data(remove_self_loop=False):
    return load_drug_data(remove_self_loop)