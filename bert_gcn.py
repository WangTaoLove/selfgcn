#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel
from gcn import GCNLayer

class SelfAttention(nn.Module):
    def __init__(self, dropout=None):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values, mask=None):
        d = queries.shape[-1]
        scores = torch.matmul(queries, keys.transpose(1, 0)) / math.sqrt(d)
        scores = scores.squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = self.softmax(scores)
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, values)
        return output, attention_weights



class BertGCN(nn.Module):
    def __init__(self, edges, features, args):
        super(BertGCN, self).__init__()
        self.label_features = features
        self.args = args
        self.edges = edges
        self.device = args.device
        self.dropout = nn.Dropout(args.dropout_prob)
        self.bert = AutoModel.from_pretrained(args.mode_path)
        self.A = nn.Parameter(torch.randn(features.size(0), features.size(0)), requires_grad=True).to(args.device)
        self.relu = nn.ReLU()  # 定义激活函数
        self.GCN = GCNLayer(features.size(1), self.bert.config.hidden_size)
        self.attention = SelfAttention(dropout=0.2)
        if args.Attention:
            self.classifier = nn.Linear(self.bert.config.hidden_size*2, self.label_features.size(0))
        else:
            self.classifier = nn.Linear(self.bert.config.hidden_size, self.label_features.size(0))
        
        
    def forward(self, input_ids, attention_mask):
        hidden_output = self.bert(input_ids, attention_mask)['last_hidden_state']
        if self.args.GCN:
            text_output = self.dropout(hidden_output[:, 0])
            edges = self.relu(self.A)
            # print('邻接矩阵参数打印：',edges)
            label_embed = self.GCN(self.label_features, edges)
            if self.args.Attention:
                att_output, attn = self.attention(text_output, label_embed, label_embed)
                pooled_output = torch.cat([text_output, att_output], dim=-1)
                bert_output = text_output
                output = self.classifier(pooled_output)
            else:
                label_embed = F.relu(label_embed)
                bert_output = text_output
                output = torch.matmul(bert_output, label_embed.T)
        else:
            bert_output = hidden_output.mean(dim=1).squeeze(1)
            output = self.classifier(bert_output)

        return output, bert_output
