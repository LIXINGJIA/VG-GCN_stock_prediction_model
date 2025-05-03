import os
import train
import  pandas as  pd
import torch
import numpy as  np

def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

files = os.listdir('\data')
kong = pd.DataFrame(columns=['股票代码', 'accuracy', 'precision', 'recall', 'F1'])
for  i in  files:
    i=i[:-4]
    # print(i)
    (accuracy, precision, recall, F1)=train.onetrain(i)
    kong = kong.append({'股票代码': i, "accuracy": accuracy, "precision": precision, 'recall': recall, 'F1': F1}, ignore_index=True)
    kong.to_csv('test.csv')

