# -*- coding:utf-8 -*-
import json
import math
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import os
import torch
from functools import partial
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
import random
from functools import partial
from sklearn.preprocessing import MultiLabelBinarizer
from utils import *


class MyDataset(object):
    
    def __init__(self,
                 args,
                 train_data_path,
                 val_data_path,
                 test_data_path,
                 tokenizer,
                 batch_size,
                 max_length,
                 sbert):
        
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sbert = sbert
        self.args = args
        self.labels_desc = json.load(open(args.data_type_path, encoding='gbk'))
        self.labels_index = {}
        self.labels = []
        
        self.train_loader, self.val_loader, self.test_loader, self.edges, self.label_features = self.load_dataset(train_data_path, val_data_path, test_data_path)
    
    def load_dataset(self, train_data_path, val_data_path, test_data_path):
        with open(train_data_path) as f:
            train = f.readlines()
        with open(val_data_path) as f:
            val = f.readlines()
        with open(test_data_path) as f:
            test = f.readlines()
        train_sents, train_labels_list = data_proc(train)
        self.labels = train_labels_list
        for index, label in enumerate(train_labels_list):
            for lab in label:
                if lab not in self.labels_index:
                    self.labels_index[lab] = [index]
                else:
                    self.labels_index[lab].append(index)

        val_sents, val_labels_list = data_proc(val)
        test_sents, test_labels_list = data_proc(test)
        print("Numbers of train: ", len(train_sents))
        print("Numbers of val: ", len(val_sents))
        print("Numbers of test: ", len(test_sents))
        mlb = MultiLabelBinarizer()

        train_labels = mlb.fit_transform(train_labels_list)
        print("Numbers of labels: ", len(mlb.classes_))
        val_labels = mlb.transform(val_labels_list)
        test_labels = mlb.transform(test_labels_list)
        edges, label_features = self.create_edges_and_features(train_labels_list, mlb)
        
        train_loader = self.encode_data(train_sents, train_labels, mode='train', shuffle=True)
        val_loader = self.encode_data(val_sents, val_labels, mode='val', shuffle=False)
        test_loader = self.encode_data(test_sents, test_labels, mode='test', shuffle=False)
        
        return train_loader, val_loader, test_loader, edges, label_features
    
    def create_edges_and_features(self, train_labels, mlb):
        label2id = {v: k for k, v in enumerate(mlb.classes_)}
        edges = torch.zeros((len(label2id), len(label2id)))
        for label in train_labels:
            if len(label) >= 2:
                for i in range(len(label) - 1):
                    for j in range(i + 1, len(label)):
                        src, tgt = label2id[label[i]], label2id[label[j]]
                        edges[src][tgt] += 1
                        edges[tgt][src] += 1
        
        marginal_edges = torch.zeros((len(label2id)))
        
        for label in train_labels:
            for i in range(len(label)):
                marginal_edges[label2id[label[i]]] += 1
        
        for i in range(edges.size(0)):
            for j in range(edges.size(1)):
                if edges[i][j] != 0:
                    edges[i][j] = (edges[i][j] * len(train_labels))/(marginal_edges[i] * marginal_edges[j])
                    

        edges = normalizeAdjacency(edges + torch.diag(torch.ones(len(label2id))))
        features = torch.zeros(len((label2id)), 768)
        for label, id in tqdm(label2id.items()):
            label_description = self.labels_desc[label]
            features[id] = get_embedding(self.sbert, label_description, n_sent=2)
            
        return edges, features
    
    def encode_data(self, train_sents, train_labels, mode, shuffle=False):
        X_train = self.tokenizer.batch_encode_plus(train_sents, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        y_train = torch.tensor(train_labels, dtype=torch.float)
        
        train_tensor = TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_train)
        if mode == 'train':
            if self.args.Contrast:
                train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True)
            else:
                train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True)
        else:
            train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=shuffle)
        
        return train_loader


