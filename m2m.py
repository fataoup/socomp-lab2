import json
import numpy as np
import pickle


def m2m():

    with open('member_index.json', 'r') as f:
        member_index = json.load(f)
    
    members = list(member_index.keys())
    members = list(set(members))
    num_members = len(members)
    print(members)
    print(num_members)

    # 产生训练集的mem2mem矩阵

    with open('train_members.json', 'r') as f:
        data = json.load(f)


    M = np.zeros((num_members, num_members))
    for item in data:
        subclub1_members, subclub1_indices, subclub2_members, subclub2_indices = [], [], [], []
        if item['subclub']==1 and set(item['members']).issubset(set(members)):
            subclub1_members = item['members']
            subclub1_indices = [members.index(m) for m in subclub1_members]
            for i in range(len(subclub1_indices)):
                for j in range(i+1, len(subclub1_indices)):
                    M[subclub1_indices[i], subclub1_indices[j]] = 1
                    M[subclub1_indices[j], subclub1_indices[i]] = 1
        elif item['subclub'] == 0 and set(item['members']).issubset(set(members)):
            subclub2_members = item['members']
            subclub2_indices = [members.index(m) for m in subclub2_members]
            for i in range(len(subclub2_indices)):
                for j in range(i+1, len(subclub2_indices)):
                    M[subclub2_indices[i], subclub2_indices[j]] = 1
                    M[subclub2_indices[j], subclub2_indices[i]] = 1

    print(M)
    print(np.sum(M))
    np.save('m2m_new.npy', M)

def norm_and_tolist():

    mat = np.load('m2m_new.npy')
    num_rows, num_cols = mat.shape

    # 对每行归一化
    normalized_mat = mat / np.sum(mat, axis=1, keepdims=True)

    # 构建列表:[src,tgt,weight]的列表
    m2m_list = []
    for i in range(num_rows):
        for j in range(num_cols):
            m2m_list.append([i, j, normalized_mat[i, j]])
    
    with open('m2mlist.pkl', 'wb') as f:
        pickle.dump(m2m_list, f)


if __name__ == '__main__':
    m2m()
    norm_and_tolist()

