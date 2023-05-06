import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def pre_process(train_file='data/train_votes.json', 
                test_file='data/test_votes.json'
                ):
    train_df = pd.read_json(train_file, orient='records')
    test_df = pd.read_json(test_file, orient='records')
    
    column_reserve = ['bill_type', 'group', 'district', 'vote']
    # column_reserve = train_df.columns

    map_dict = {}
    for column in train_df.columns:
        if column in column_reserve:
            factorized_column, map = train_df[column].factorize()
            train_df[column] = factorized_column
            map_dict[column] = map
        else:
            train_df = train_df.drop(columns=column)

    for column in test_df.columns:
        # factorized_column, _ = test_df[column].factorize()
        # test_df[column] = factorized_column
        if column in column_reserve:
            dict = {map_dict[column][i]:i for i in range(len(map_dict[column]))}

            test_df[column] = test_df[column].replace(dict)
        else:
            test_df = test_df.drop(columns=column)

    train_features, train_label = train_df.iloc[:,:-1], train_df.iloc[:,-1]
    test_features, test_label = test_df.iloc[:,:-1], test_df.iloc[:,-1]

    return (train_features, train_label), (test_features, test_label), map_dict

def fit(features, label):
    clf = CategoricalNB()
    clf.fit(features, label)
    return clf


if __name__ == '__main__':
    (train_features, train_label), (test_features, test_label), map_dict = pre_process()
    print('start training bayes...')
    clf = fit(train_features, train_label)
    y_pred = clf.predict(test_features)

    # computing results including precision, recall, f1score, accuracy
    precision, recall, f1score, _ = precision_recall_fscore_support(test_label, y_pred)
    acc = accuracy_score(test_label, y_pred)
    print("precision:{} \nrecall:{} \nf1score:{} \naccuracy:{} \n".format(precision, recall, f1score, acc))

    # output csv
    # rev_dict = {index:map_dict[10][index] for index in range(len(map_dict[10]))}
    # pred_income = [rev_dict[i] for i in y_pred]
    # test_df = pd.read_csv('dataset/Bayesian_Dataset_test.csv', header=None)
    # test_df.insert(len(test_df.columns), 11, pred_income)
    # test_df.to_csv('output/Testset_with_predicted_income.csv', header=None, index=None)
