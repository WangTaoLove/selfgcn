import re
import math
import random, os
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch

def data_proc(context):
    labels = []
    data = []
    for content in context:
        content = content.strip('\n')
        label = content.split('\t')[0]
        data.append(content.split('\t')[1])
        labels.append(label.split('|'))
    return data, labels

    
def normalizeAdjacencyv2(W):
    assert W.size(0) == W.size(1)
    d = torch.sum(W, dim = 1)
    d = 1/d
    D = torch.diag(d)
    return D @ W 

def normalizeAdjacency(W):
    assert W.size(0) == W.size(1)
    d = torch.sum(W, dim = 1)
    d = 1/torch.sqrt(d)
    D = torch.diag(d)
    return D @ W @ D 

def get_embedding(sbert, text, n_sent=1):
    embedding = sbert.encode(text, convert_to_tensor=True)
    return embedding

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



class FGM():
    def __init__(self, model, epsilon=0.5, ):
        # BERT模型
        self.model = model
        # 求干扰时的系数值
        self.epsilon = epsilon

        self.backup = {}

    def attack(self, emb_name='word_embeddings'):
        # 循环遍历模型所有参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息，并且包含了模型中embedding的参数名
            if param.requires_grad and emb_name in name:
                # 把真实参数保存起来
                self.backup[name] = param.data.clone()
                # 对参数的梯度求范数
                norm = torch.norm(param.grad)
                # 如果范数不等于0并且norm中没有缺失值
                if norm != 0 and not torch.isnan(norm):
                    # 计算扰动，param.grad / norm=单位向量，起到了sgn(param.grad)一样的作用
                    r_at = self.epsilon * param.grad / norm
                    # 在原参数的基础上添加扰动
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # 循环遍历模型所有参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息，并且包含了模型中embedding的参数名
            if param.requires_grad and emb_name in name:
                # 断言
                assert name in self.backup
                # 取出模型真实参数
                param.data = self.backup[name]
        # 清空self.backup
        self.backup = {}