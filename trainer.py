#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer,get_constant_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from bert_gcn import BertGCN
from dataset import MyDataset
from losses import criterion
from sklearn import metrics
from torchmetrics import Precision
from tqdm import tqdm
from utils import FGM


class TrainerModel(object):
    def __init__(self, args):
        self.args = args
        self.sbert = SentenceTransformer(args.sbert, device='cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        self.dataset = MyDataset(args,
                                 args.train_data_path,
                                 args.val_data_path,
                                 args.test_data_path,
                                 self.tokenizer,
                                 args.batch_size,
                                 args.max_length,
                                 self.sbert)

        self.train_loader, self.val_loader, self.test_loader = self.dataset.train_loader, self.dataset.val_loader, self.dataset.test_loader

        self.model = BertGCN(self.dataset.edges.to(args.device),
                             self.dataset.label_features.to(args.device),
                             args)
        self.save_mode = f"bert.bin"
        if args.Contrast:
            self.save_mode = 'Contrast_' + self.save_mode
        if args.GCN:
            self.save_mode = 'GCN_' + self.save_mode
        if args.Attention:
            self.save_mode = 'Attention_' + self.save_mode
        if self.args.pre_trained:
            self.model.load_state_dict(torch.load(f'./output/{self.save_mode}', map_location=args.device))
        self.model.to(args.device)
        self.adv_trainer = FGM(self.model)
        self.optimizer, self.scheduler = self._get_optimizer()
        self.criterion_loss = criterion(args)

    def _get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'layer_norm.bias', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        if self.args.Contrast:
            optimizer = AdamW(optimizer_grouped_parameters, lr=0.00002)
            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                        num_training_steps=self.args.n_epochs * len(self.train_loader),
                                                        num_warmup_steps=100)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_training_steps=self.args.n_epochs * len(self.train_loader),
                                                    num_warmup_steps=100)

        return optimizer, scheduler

    def validate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            # total_loss = 0.0
            predicted_labels, target_labels = list(), list()
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask, y_true = tuple(t.to(self.args.device) for t in batch)
                output, embedding = self.model.forward(input_ids, attention_mask)
                target_labels.extend(y_true.cpu().detach().numpy())
                predicted_labels.extend(torch.sigmoid(output).cpu().detach().numpy())

        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels.round())
        micro_recall = metrics.recall_score(target_labels, predicted_labels.round(), average='micro')
        micro_precision = metrics.precision_score(target_labels, predicted_labels.round(), average='micro')
        micro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='micro')
        macro_recall = metrics.recall_score(target_labels, predicted_labels.round(), average='macro')
        macro_precision = metrics.precision_score(target_labels, predicted_labels.round(), average='macro')
        macro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='macro')

        return micro_f1, macro_f1, micro_recall, micro_precision, macro_recall, macro_precision, accuracy

    def train(self):
        print("******Training...******")
        best_score = float("-inf")
        for epoch in range(self.args.n_epochs):
            total_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                self.model.train()
                input_ids, attention_mask, label = tuple(t.to(self.args.device) for t in batch)
                self.optimizer.zero_grad()
                logits1, embedding1 = self.model.forward(input_ids, attention_mask)
                if self.args.Contrast:
                    logits2, embedding2 = self.model.forward(input_ids, attention_mask)
                else:
                    logits2 = None
                    embedding2 = None
                loss = self.criterion_loss(logits1, logits2, embedding1, embedding2, label)
                loss.backward()
                # if self.args.Contrast:
                #     self.adv_trainer.attack()
                #     logits, embedding = self.model.forward(input_ids, attention_mask)
                #     loss_at = nn.BCEWithLogitsLoss()(logits, label)
                #     loss_at.backward()
                #     self.adv_trainer.restore()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                if (i + 1) % 50 == 0 or i == 0 or i == len(self.train_loader) - 1:
                    print("Epoch: {} - iter: {}/{} - train_loss: {}".format(epoch, i + 1, len(self.train_loader),
                                                                            total_loss / (i + 1)))
                if i == len(self.train_loader) - 1 or (i + 1) % 100 == 0:
                    print("***************Evaluating...*******************")
                    micro_f1, macro_f1, micro_recall, micro_precision, macro_recall, macro_precision, accuracy = self.validate(
                        self.val_loader)
                    print("Accuracy:{} -Micro_F1: {} -micro_recall: {} - micro_precision: {} ".format(
                        accuracy,
                        micro_f1,
                        micro_recall,
                        micro_precision))
                    print("Macro-F1: {}- macro_recall: {} - macro_precision: {}".format(macro_f1,
                                                                                        macro_recall,
                                                                                        macro_precision))
                    if best_score <= micro_f1:
                        best_score = micro_f1
                        torch.save(self.model.state_dict(), f'./output/{self.save_mode}')
                        print('*********************save model********************')

    def test(self):
        print("Testing...")
        self.model.load_state_dict(torch.load(f'./output/{self.save_mode}', map_location=self.args.device))
        micro_f1, macro_f1, micro_recall, micro_precision, macro_recall, macro_precision, accuracy = self.validate(
            self.test_loader)
        print("Accuracy:{} -Micro_F1: {} -micro_recall: {} - micro_precision: {} ".format(
            accuracy,
            micro_f1,
            micro_recall,
            micro_precision))
        print("Macro-F1: {}- macro_recall: {} - macro_precision: {}".format(macro_f1,
                                                                            macro_recall,
                                                                            macro_precision))
