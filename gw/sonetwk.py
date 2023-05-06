import json
import random
import numpy as np

import dgl

def votes():
    with open('/data/lygao/dataset/socomp/process_data/votes.json', 'r') as f:
        data    = f.read()
    votes_data  = json.loads(data)  
    
    with open('/data/lygao/dataset/socomp/process_data/cosponsors.json', 'r') as f:
        data    = f.read()
    bill_data   = json.loads(data)
    
    member_id   = set()
    bill_detail = {}    #{billname:{type, sponsor, cosponsor, Ymember, Nmember, NVmember}}
    member  = {}        #{member:{billname, group, district, coop}}
    
    for vote in votes_data:
        name    = vote['bill_name']
        id  = vote['id']
        member_id.add(id)
        
        if id not in member.keys():
            member[id]  = {'vote':set(), 'group':vote['group'], 'district':vote['district'], 'coop':set()}
        member[id]['vote'].add(name)
            
        if name not in bill_detail.keys():
            bill_detail[name]   = {'type':vote['bill_type'], 'Y':set(), 'N':set(), 'NV':set()}
        option  = vote['vote']
        bill_detail[name][option].add(id)
        
    for bill in bill_data:
        name    = bill['bill_id']
        sponsors    = bill['cosponsor'].append(bill['sponsor'])
        member_id.update(sponsors)
        
        for i in sponsors:
            member[i]['coop'].update(sponsors)
        
        if name in bill_detail.keys():
            bill_detail[name]['sponsor']    = bill['sponsor']
            bill_detail[name]['cosponsor']  = bill['cosponsor']
            
    return bill_detail, member, list(member_id)

def get_sonetwk(bill_detail, member, member_id):
    # 时间无关
    sonetwk = np.zeros(len(member_id), len(member_id))
    member2id   = {}
    for step, id in enumerate(member_id):
        member2id[id]=step
        
    #相同加1，不同减1
    for bill in bill_detail.keys():
        detail  = bill_detail[bill]
        
        for i in detail['Y']:#set
            for j in detail['Y']:
                i_id    = member2id[i]
                j_id    = member2id[j]
                sonetwk[i_id][j_id]    += 1
                   
        for i in detail['N']:#set
            for j in detail['N']:
                i_id    = member2id[i]
                j_id    = member2id[j]
                sonetwk[i_id][j_id]    += 1
        
        for i in detail['Y']:#set
            for j in detail['N']:
                i_id    = member2id[i]
                j_id    = member2id[j]
                sonetwk[i_id][j_id]    -= 1
                
        for i in detail['N']:#set
            for j in detail['Y']:
                i_id    = member2id[i]
                j_id    = member2id[j]
                sonetwk[i_id][j_id]    -= 1
                
    return sonetwk

def get_relationship(sonetwk, member_id, member):
    for i,_ in enumerate(sonetwk):
        for j,_ in enumerate(i):
            i_name  = member_id[i]
            j_name  = member_id[j]
            
            muti    = 1
            if member[i]['group'] == member[j]['group']:
                muti *= 1.5
            if member[i]['district'] == member[j]['district']:
                muti *= 2
            if i_name in member[j]['coop']:
                muti *= 2
            
            sonetwk[i][j] *= muti
    return sonetwk

