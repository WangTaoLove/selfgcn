# -*- coding:utf-8 -*-
import logging, time, sys, os
sys.path.append("..")
from argparse import ArgumentParser
from trainer import TrainerModel
import torch
from utils import seed_everything
import warnings
warnings.filterwarnings("ignore")
seed_everything(42)

if __name__ == '__main__':
    parser = ArgumentParser(description="Model trainer")
    parser.add_argument("--train_data_path", type=str, default=f'./data/reduce_data/multi_label_train.txt')
    parser.add_argument("--val_data_path", type=str, default=f'./data/reduce_data/multi_label_test.txt')
    parser.add_argument("--test_data_path", type=str, default=f'./data/reduce_data/multi_label_test.txt')
    parser.add_argument("--data_type_path", type=str, default=f'./data/reduce_data/type_description.json')
    parser.add_argument("--mode_path", type=str, default='./bert-base-case',
                        help="Pretrained Transformer Model")
    parser.add_argument("--tokenizer_name", type=str, default='./bert-base-case')
    parser.add_argument("--sbert", type=str, default='./bert-base-case')
    parser.add_argument("--mode", type=str, default='train',
                        help="train/test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument('--Contrast', default=False, type=bool, help='Contrastive Learning')
    parser.add_argument('--GCN', default=False, type=bool, help='GCN of lable')
    parser.add_argument('--Attention', default=False, type=bool, help='attention')
    parser.add_argument("--learning_rate", type=float, default=0.00003)
    parser.add_argument("--dropout_prob", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.015)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--pre_trained', default=False, type=bool, help='load pre_trained model')
    parser.add_argument('--kl_weight', default=0.1, type=float, help='KLDivLoss weight')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for contrast loss function')
    args = parser.parse_args()
    print(f"hyper-parameters:{args}")
    """select GPU"""
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
        print("*****GPU加载成功*****")
    else:
        args.device = torch.device("cpu")
    model_trainer = TrainerModel(args)

    if args.mode == "train":
        model_trainer.train()
        print("************test*************")
        model_trainer.test()
    elif args.mode == "test":
        model_trainer.test()
