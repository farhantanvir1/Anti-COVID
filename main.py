#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:58:38 2021

@author: illusionist
"""
# execute
#pip install dgl-cu101
# main.py
import torch
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_recall_curve

import numpy as np
import csv

from util import load_data
from csv import DictReader
import numpy as np
import networkx as nx
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
from model import GNN
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_recall_curve, confusion_matrix
"""
    this function creates a dictionary for entity(drug/protein/species/indication) sent as argument
"""

def create_dictionary(path):
    
    count=1
    dict_val={}
    # dictionary's key will content entity object's id(drug id/protein id)
    # value will contain the index id where corresponding entity object's data will be found in adjacency and
    # commuting matrix
    with open(path) as f:
        content = f.readlines()
        for x in content:
            dict_val[x.strip()]=count
            count+=1
    
    
    return dict_val

def score(logits, labels):

    accuracy = (y_predict == y_test).sum() / len(y_predict)
    micro_f1 = f1_score(y_test, y_predict, average='micro')
    print("micro f1: %.2f%%" % (micro_f1 * 100.0)) 
    macro_f1 = f1_score(y_test, y_predict, average='macro')
    print("macro f1: %.2f%%" % (macro_f1 * 100.0))
    #roc = roc_auc_score(y_test, y_predict, multi_class="ovr")
    #print("roc: %.2f%%" % (roc * 100.0)) 
    rec = recall_score(y_test, y_predict, average='macro')
    print("recall: %.2f%%" % (rec * 100.0)) 
    prc = precision_score(y_test, y_predict, average='macro')
    print("precision: %.2f%%" % (prc * 100.0)) 
    #aupr = average_precision_score(y_test, y_predict, average='macro')
    #print("AUPR: %.2f%%" % (aupr * 100.0))

    return accuracy, micro_f1, macro_f1, rec, prc 

def main():
    g, labels, Compound_dict, Disease_dict, train_arr, test_arr = load_data()
    temp=labels
    print('labels',labels.shape)
    A = nx.adjacency_matrix(g)

    
    # FT 110621 get node embeddings based on GCN
    modelGCN = GNN(A, supervised=False, model='gcn', device='cuda')
    gcn_embedding = modelGCN.generate_embeddings()
    modelGCN.fit()
    # get the node embeddings with the trained models
    gcn_embedding = modelGCN.generate_embeddings()

    np.save("/content/drive/MyDrive/Anti-COVID (1)/data/gcn_embedding.npy", gcn_embedding)

    # 112521 get node embeddings based on GS
    modelGS = GNN(A, supervised=False, model='graphsage', device='cuda')
    gs_embedding = modelGS.generate_embeddings()
    modelGS.fit()
    # get the node embeddings with the trained models
    gs_embedding = modelGS.generate_embeddings()

    np.save("/content/drive/MyDrive/Anti-COVID (1)/data/gs_embedding.npy", gs_embedding)
    
    # 112621 get node embeddings based on gat
    modelGAT = GNN(A, supervised=False, model='gat', device='cuda')
    gat_embedding = modelGAT.generate_embeddings()
    modelGAT.fit()
    # get the node embeddings with the trained models
    gat_embedding = modelGAT.generate_embeddings()
    
    np.save("/content/drive/MyDrive/Anti-COVID (1)/data/gat_embedding.npy", gat_embedding)
    
    
    gat_embedding = np.load('/content/drive/MyDrive/Anti-COVID (1)/data/gat_embedding.npy')
    print(type(gat_embedding),gat_embedding.shape)

    gcn_embedding = np.load('/content/drive/MyDrive/Anti-COVID (1)/data/gcn_embedding.npy')
    print(type(gcn_embedding),gcn_embedding.shape)

    gs_embedding = np.load('/content/drive/MyDrive/Anti-COVID (1)/data/gs_embedding.npy')
    print(type(gs_embedding),gs_embedding.shape)

    gcn_features_train=[[]]
    gcn_features_test=[[]]
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows_train, cols_train= (62500, 131)
    rows_test, cols_test= (15626, 131)

    gcn_features_train = [[0] * cols_train for i in range(rows_train)]
    gcn_features_test = [[0] * cols_test for i in range(rows_test)]
    incr=9572
        
    for i in range(len(train_arr)):
            a=int(train_arr[i][0])
            b=int(train_arr[i][1])
            gcn_features_train[i][0]=a
            gcn_features_train[i][1]=b
            gcn_features_train[i][2:66]=gcn_embedding[a]
            gcn_features_train[i][66:130]=gcn_embedding[incr+b]
            gcn_features_train[i][130]=labels[a][b]


    for i in range(len(test_arr)):
            a=int(test_arr[i][0])
            b=int(test_arr[i][1])
            gcn_features_test[i][0]=a
            gcn_features_test[i][1]=b
            gcn_features_test[i][2:66]=gcn_embedding[a]
            gcn_features_test[i][66:130]=gcn_embedding[incr+b]
            gcn_features_test[i][130]=labels[a][b]

    np.save("/content/drive/MyDrive/Anti-COVID (1)/data/gcn_features_train.npy", gcn_features_train)
    np.save("/content/drive/MyDrive/Anti-COVID (1)/data/gcn_features_test.npy", gcn_features_test)
    print('GCN is done!')

    gs_features_train=[[]]
    gs_features_test=[[]]
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows_train, cols_train= (62500, 131)
    rows_test, cols_test= (15626, 131)

    gs_features_train = [[0] * cols_train for i in range(rows_train)]
    gs_features_test = [[0] * cols_test for i in range(rows_test)]
    incr=9572
        
    for i in range(len(train_arr)):
            print()
            a=int(train_arr[i][0])
            b=int(train_arr[i][1])
            gs_features_train[i][0]=a
            gs_features_train[i][1]=b
            gs_features_train[i][2:66]=gs_embedding[a]
            gs_features_train[i][66:130]=gs_embedding[incr+b]
            gs_features_train[i][130]=labels[a][b]


    for i in range(len(test_arr)):
            a=int(test_arr[i][0])
            b=int(test_arr[i][1])
            gs_features_test[i][0]=a
            gs_features_test[i][1]=b
            gs_features_test[i][2:66]=gs_embedding[a]
            gs_features_test[i][66:130]=gs_embedding[incr+b]
            gs_features_test[i][130]=labels[a][b]

    np.save("/content/drive/MyDrive/Anti-COVID (1)/data/gs_features_train.npy", gs_features_train)
    np.save("/content/drive/MyDrive/Anti-COVID (1)/data/gs_features_test.npy", gs_features_test)
    print('GS is done!')


    gat_features_train=[[]]
    gat_features_test=[[]]
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows_train, cols_train= (62500, 131)
    rows_test, cols_test= (15626, 131)

    gat_features_train = [[0] * cols_train for i in range(rows_train)]
    gat_features_test = [[0] * cols_test for i in range(rows_test)]
    incr=9572
        
    for i in range(len(train_arr)):
            a=int(train_arr[i][0])
            b=int(train_arr[i][1])
            gat_features_train[i][0]=a
            gat_features_train[i][1]=b
            gat_features_train[i][2:66]=gat_embedding[a]
            gat_features_train[i][66:130]=gat_embedding[incr+b]
            gat_features_train[i][130]=labels[a][b]


    for i in range(len(test_arr)):
            a=int(test_arr[i][0])
            b=int(test_arr[i][1])
            gat_features_test[i][0]=a
            gat_features_test[i][1]=b
            gat_features_test[i][2:66]=gat_embedding[a]
            gat_features_test[i][66:130]=gat_embedding[incr+b]
            gat_features_test[i][130]=labels[a][b]

    np.save("/content/drive/MyDrive/Anti-COVID (1)/data/gat_features_train.npy", gat_features_train)
    np.save("/content/drive/MyDrive/Anti-COVID (1)/data/gat_features_test.npy", gat_features_test)
    print('GAT is done!')
    
if __name__ == '__main__':
    main()