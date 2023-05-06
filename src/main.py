import os
import torch
import torch.nn as nn
import dgl
import numpy as np
from tqdm import tqdm
import json
from modules import SAGEMLP

vote_map = {'N': 0, 'NV': 1, 'Y': 2}

def build_graph(vote_file='data/votes.json'):
    graph_mat = []
    train_mask = []
    test_mask = []
    labels = []
    member_map = {}
    bill_map = {}
    num_members = 0
    num_bills = 0

    with open(vote_file) as f:
        vote_list = json.load(f)
        for vote in tqdm(vote_list):
            if vote['bill_name'] not in bill_map.keys():
                bill_map[vote['bill_name']] = num_bills
                num_bills += 1
            if vote['id'] not in member_map.keys():
                member_map[vote['id']] = num_members
                num_members += 1

            graph_mat.append([member_map[vote['id']], bill_map[vote['bill_name']]])
            labels.append(vote_map[vote['vote']])

            if vote['bill_name'][-3:-1] == '10' or vote['bill_name'][-3:] in ['110', '111', '112', '113']:
                train_mask.append(1)
                test_mask.append(0)
            else:
                train_mask.append(0)
                test_mask.append(1)

    graph_mat = np.array(graph_mat)
    print('total bills :{}, total members :{}'.format(num_bills, num_members))

    return graph_mat, train_mask, test_mask, labels, num_members, num_bills


def train(model, graph, feature, train_mask, num_epochs=1000):

    # train
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    labels = graph.edata['labels']
    for epoch in range(num_epochs):
        model.train()
        # node_embed = sage_gcn(graph, feature)
        logits = model(graph, feature)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc, indices = evaluate(model, graph, feature)
            print('Epoch {} :train loss: {:.5f}, acc: {:.5f}'.format(epoch, loss.item(), acc))
            # print(indices, torch.sum(indices))

def evaluate(model, graph, feature):
    model.eval()
    eval_mask = graph.edata['test_mask']
    labels = graph.edata['labels']
    with torch.no_grad():
        logits = model(graph, feature)
        logits = logits[eval_mask]
        labels = labels[eval_mask]

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), indices



if __name__ == '__main__':
    device = 'cuda:7'
    feature_dim = 32
    graph_path = 'output/graph.bin'
    if not os.path.exists(graph_path):
    # if True: 
        graph_mat, train_mask, test_mask, labels, num_members, num_bills = build_graph()
        graph_mat[:,1] += num_members
        graph = dgl.graph((torch.tensor(graph_mat[:,0]), torch.tensor(graph_mat[:,1])))
        
        graph.edata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
        graph.edata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)
        graph.edata['labels'] = torch.tensor(labels, dtype=torch.long)

        graph.ndata['feat'] = torch.randn((graph.num_nodes(), feature_dim))
        dgl.save_graphs(graph_path, graph)
    else:
        
        graph = dgl.load_graphs(graph_path)[0][0]

    sage_mlp = SAGEMLP(feature_dim, 3).to(device)
    feature = graph.ndata['feat'].to(device)
    # graph = dgl.to_bidirected(graph)
    graph = graph.to(device)
    train_mask = graph.edata['train_mask']

    train(sage_mlp, graph, feature, train_mask)

    pass