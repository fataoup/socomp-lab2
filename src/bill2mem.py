import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# 产生训练集的bill2mem矩阵

with open('data/train_votes.json', 'r') as f:
    votes = json.load(f)

bills=[]
for vote in votes:
    bills.append(vote['bill_name'])

bills = list(set(bills)) # 去重
num_bills = len(bills) # 训练集包含的member数
print(num_bills)


with open('train_members.json', 'r') as f:
    mems = json.load(f)

members=[]
for mem in mems:
    members.extend(mem['members'])
members = list(set(members)) # 去重
num_members = len(members) # 训练集包含的member数
print(num_members)

M = np.zeros((num_bills, num_members))

for vote in tqdm(votes):
    if vote['bill_name'] in bills and vote['id'] in members:
        i = bills.index(vote['bill_name'])
        j = members.index(vote['id'])
    if vote['vote'] == 'Y':
        M[i][j] = 1
    elif vote['vote'] == 'N':
        M[i][j] = -1
    elif vote['vote'] == 'NV':
        M[i][j] = 0
print(M)
np.savetxt('bill2mem.txt', M, delimiter=',')

