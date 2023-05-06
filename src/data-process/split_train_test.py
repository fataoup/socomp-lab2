import json
from tqdm import tqdm

vote_file = 'data/votes.json'

train_data = []
test_data = []

with open(vote_file, 'r') as f, open('data/train_votes.json', 'w+') as fw_train, open('data/test_votes.json', 'w+') as fw_test:
    data = json.load(f)
    for item in tqdm(data):
        if item['bill_name'][-3:-1] == '10' or item['bill_name'][-3:] in ['110', '111', '112', '113']:
            train_data.append(item)
        else:
            test_data.append(item)

    # json.dump(train_data, fw_train)
    # json.dump(test_data, fw_test)