import json
import random
import numpy as np
from sklearn.preprocessing import normalize
import dgl

def votes():
    with open('/data/lygao/dataset/socomp/process_data/votes.json', 'r') as f:
        data    = f.read()
    votes_data  = json.loads(data)  
    
    with open('/data/lygao/dataset/socomp/process_data/cosponsors.json', 'r') as f:
        data    = f.read()
    bill_data   = json.loads(data)
    
    member_id   = set()
    member2id   = {}
    bill_detail = {}    #{billname:{type, sponsor, cosponsor, Ymember, Nmember, NVmember}}
    member  = {}        #{member:{billname, group, district, coop}}
    
    for vote in votes_data:
        name    = vote['bill_name']
        id  = vote['id']
        member_id.add(id)
    member_id   = list(member_id)    
    
    for step, id in enumerate(member_id):   #change member id to number
        member2id[id]=step
    
    for vote in votes_data: 
        name    = vote['bill_name']
        member_id   = vote['id'] 
        id  = member2id[member_id]  
        
        if id not in member.keys():
            member[id]  = {'vote':set(), 'group':vote['group'], 'district':vote['district'], 'coop':set()}
        member[id]['vote'].add(name)
            
        if name not in bill_detail.keys():
            bill_detail[name]   = {'type':vote['bill_type'], 'Y':set(), 'N':set(), 'NV':set()}
        option  = vote['vote']
        bill_detail[name][option].add(id)
        
    for bill in bill_data:
        name    = bill['bill_id']
        sponsors    = []   # 一些sponsors不在投票者记录中，不参与构建
        
        for i in bill['cosponsors']:
            if i in member2id.keys():
                sponsors.append(member2id[i])              
        if  bill['sponsor'] in member2id.keys():    
            sponsors.append(member2id[bill['sponsor']])
        #member_id.update(sponsors)
        
        for i in sponsors:
            member[i]['coop'].update(sponsors)
        
        if name in bill_detail.keys():
            bill_detail[name]['sponsor']    = bill['sponsor']
            bill_detail[name]['cosponsors'] = bill['cosponsors']
            
    return bill_detail, member, member_id, member2id

def get_sonetwk(bill_detail, member, member2id):
    # 时间无关
    sonetwk = np.zeros((len(member2id), len(member2id)))

    #相同加1，不同减1
    for bill in bill_detail.keys():
        detail  = bill_detail[bill]
        
        for i in detail['Y']:#set
            for j in detail['Y']:
                sonetwk[i][j]    += 1
                   
        for i in detail['N']:#set
            for j in detail['N']:
                sonetwk[i][j]    += 1
        
        for i in detail['Y']:#set
            for j in detail['N']:
                sonetwk[i][j]    -= 1
                sonetwk[j][i]    -= 1
                
    return sonetwk

def get_relationship(snk1, member):
    for i,a in enumerate(snk1):
        for j,_ in enumerate(a):    
            muti    = 1
            if member[i]['group'] == member[j]['group']:
                muti *= 1.5
            if member[i]['district'] == member[j]['district']:
                muti *= 2
            if i in member[j]['coop']:
                muti *= 2           
            snk1[i][j] *= muti
    snk1    = np.rint(snk1)
    return snk1

def mlpinfeats(sonetwk, bill_detail, member, member2id):# get from each votes
    infeats = []
    labels  = []
    for bill_name in bill_detail.keys():    #need cacl looong time
        bill    = bill_detail[bill_name]
        billfeat    = []
        billfeat.append(bill_name[-3:])
        billfeat.append(bill['type'])
        
        if bill['sponsor']!=-1:
            sponsor = bill['sponsor']
            billfeat.append(member[sponsor]['group'])
            billfeat.append(member[sponsor]['district'])
        else:
            billfeat.append(-1) 
            billfeat.append(-1) 

        for memid in bill['Y'] or bill['N'] or bill['NV']:
            memfeat = []
            mem = member[memid]
            memfeat.append(mem['group'])
            memfeat.append(mem['district'])
            memfeat.append(len(mem['billname']))
            
            infeats.append(memfeat+billfeat)
            if memid in bill['Y']:
                labels.append(1)
            elif memid in bill['N']:
                labels.append(-1)
            elif memid in bill['NV']:
                labels.append(0)
    return infeats, labels

def mem_idx():
    with open('/data/lygao/git/1_23/socomp-lab2/gw/member_index.json', 'r') as f:
        data    = f.read()
    name2id  = json.loads(data)  
    return name2id

def relation_matrix(snk1_n):
    sonetwk[sonetwk<2500]   = 0
    sonetwk = normalize(sonetwk, norm='l2')
    matrix  = [[],[]]
    weight  = []  
    for i, a in enumerate(snk1_n):
        for j,_ in enumerate(a):
            if snk1_n[i][j]>0:         
                matrix[0].append(i)
                matrix[1].append(j)
                weight.append(snk1_n[i][j])                 
    matrix  = np.array(matrix)
    weight  = np.array(weight)       
    return matrix, weight

