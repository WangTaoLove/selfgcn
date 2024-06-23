#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class criterion(nn.Module):
    def __init__(self, config):
        super(criterion, self).__init__()
        self.temperature = 200
        self.config = config
        self.ce = nn.BCEWithLogitsLoss()
        self.kld = nn.KLDivLoss(reduction='none')
        self.sup_con_loss = SupConLoss(config, self.temperature)

    def forward(self, logits1=None, logits2=None, embedding1=None, embedding2=None, labels=None):
        if self.config.Contrast:
            ce_loss = (self.ce(logits1, labels) + self.ce(logits2, labels)) / 2
            kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).mean()
            kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).mean()
            kl_loss = (kl_loss1 + kl_loss2) / 2
            # cumulative_loss =(self.sup_con_loss(embedding1, labels) + self.sup_con_loss(embedding2, labels))/2
            embedding = torch.cat([embedding1, embedding2], dim=0)
            labels_new = torch.cat([labels, labels], dim=0)
            cumulative_loss = self.sup_con_loss(embedding, labels_new)
            loss = ce_loss + self.config.alpha * cumulative_loss + self.config.kl_weight * kl_loss
        else:
            loss = self.ce(logits1, labels)

        return loss


class SupConLoss(nn.Module):
    def __init__(self, config, temperature=100):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.config = config

    def forward(self, features, labels=None):

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 2 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)

        batch_size = features.shape[0]
        # labels = labels.unsqueeze(-1)  # torch.Size([4, 1])
        # torch.eq,对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。
        # mask = torch.eq(labels, labels.transpose(0, 1))  # torch.Size([4, 4])
        mask = torch.matmul(labels, labels.T)  # torch.Size([16, 637])
        mask = mask.type(torch.uint8)  # 转换成数值类型
        one = torch.ones_like(mask)
        mask = mask ^ torch.diag_embed(torch.diag(mask))  # 对角线定义为0
        mask_labels = torch.where(mask > 1, one, mask)
        # # delete diag elem,torch.diag:取对角线元素
        # mask = mask.type(torch.uint8)  # 转换成数值类型
        # mask_labels = mask ^ torch.diag_embed(torch.diag(mask))  # 对角线定义为0
        anchor_feature = features
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, features.T),
            self.temperature)
        # anchor_dot_contrast = anchor_dot_contrast - torch.diag_embed(torch.diag(anchor_dot_contrast))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(
            torch.ones_like(mask_labels),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.config.device),
            0
        )
        mask_labels = mask_labels * logits_mask  # 把mask对角线的数据设置为0
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # 计算除了对角线所有的概率
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive,sum(1)每列求和
        mean_log_prob_pos = (mask_labels * log_prob).sum(1) / (mask_labels.sum(1)+1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss
