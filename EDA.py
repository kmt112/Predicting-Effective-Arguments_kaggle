# %%
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
df = pd.read_csv (r'C:\Users\tan_k\Predicting Effective Arguments_kaggle\train.csv')
# print(df.head())

# merge discourse type with main corpus of texts
# %%
df['combined'] = df['discourse_text'].astype(str) + ' [SEP] ' + df['discourse_type'].astype(str)
df = df.drop(['discourse_id', 'essay_id', 'discourse_type', 'discourse_text'], axis=1)

# %%
x = df['combined']
y = df['discourse_effectiveness']
train_df, test_df = train_test_split(x,y,stratify = y, test_size = 0)
# %%
