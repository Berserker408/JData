# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:27:12 2017

@author: lenovo
"""

from sklearn.cross_validation  import train_test_split
import xgboost as xgb
from gen_feat_zyg import reportnew
import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt  

                   
def xgboost_submission():
    refer_date_list = ["2016-04-01", "2016-04-02", "2016-04-03", "2016-04-04", "2016-04-05", 
                   "2016-04-06", "2016-04-07", "2016-04-08", "2016-04-09", "2016-04-10"]
    data_total = pd.DataFrame()
    for refer_date in refer_date_list:
        dump_path = '../cache/trainset_%s.pkl' % refer_date
        data = pickle.load(open(dump_path, 'rb'))
        data_part1 = data[data['label']==1].copy()
        data_part0 = data[data['label']==0].copy()
        _, data_part0_p, _, _ = train_test_split(data_part0, data_part0['label'], test_size=(50000-len(data_part1)), random_state=0)
        data_boot = pd.concat([data_part1, data_part0_p], axis=0)
        data_total = pd.concat([data_total, data_boot], axis=0)

    

    label = data_total['label'].copy()
    user_index = data_total[['user_id', 'sku_id']].copy()
    del data_total[['label','user_id', 'sku_id']]
    training_data  = data_total.copy()
    
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3, 
        'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
        'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic',
        'eval_metric': 'auc'}
    num_round = 300
    param['nthread'] = 4
    #param['eval_metric'] = "auc"
    plst = param.items()
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train(plst, dtrain, num_round, evallist)

    refer_date = '2016-04-15'
    dump_path = '../cache/testset_%s.pkl' % refer_date
    data_sub = pickle.load(open(dump_path, 'rb'))
    sub_user_index = data_sub[['user_id', 'sku_id']].copy()
    del data_sub['user_id']
    del data_sub['sku_id']

    sub_trainning_data  = data_sub.copy()
    
    sub_trainning_data = xgb.DMatrix(sub_trainning_data)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
#opt1
    pred = sub_user_index[sub_user_index['label'] >= 0.35]
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    print('opt1: %g' %len(pred))

#opt2
#    pred = sub_user_index.copy()
#    pred['label'] = y
#    predunique = pred.groupby('user_id').max().reset_index().copy()
#    predunique = predunique.sort_values(by='label',ascending=False)
#    del pred['label']
#    predfinal = predunique.reset_index(drop=True).copy() 
#    predfinal['label'] = (predfinal['label']>=0.03)*1  
#    pred = pd.merge(pred, predfinal, how='left', on=['user_id', 'sku_id'])
#    pred = pred.fillna(0)
#    pred = pred[pred['label'] == 1]
#    pred = pred[['user_id', 'sku_id']]

    #############
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('../sub/submission'+time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))+'.csv', index=False, index_label=False)



def xgboost_test():
    refer_date_list_1 = ["2016-04-01", "2016-04-02", "2016-04-03", "2016-04-04", "2016-04-05", 
                   "2016-04-06", "2016-04-07", "2016-04-09", "2016-04-10"]
    data_total = pd.DataFrame()
    for refer_date in refer_date_list_1:
        dump_path = '../cache/trainset_%s.pkl' % refer_date
        data = pickle.load(open(dump_path, 'rb'))
        data_part1 = data[data['label']==1].copy()
        data_part0 = data[data['label']==0].copy()
        _, data_part0_p, _, _ = train_test_split(data_part0, data_part0['label'], test_size=(50000-len(data_part1)), random_state=0)
        data_boot = pd.concat([data_part1, data_part0_p], axis=0)
        data_total = pd.concat([data_total, data_boot], axis=0)

        
        
    
    label = data_total['label'].copy()
    user_index = data_total[['user_id', 'sku_id']].copy()
    del(data_total['label'])
    del(data_total['user_id'])
    del(data_total['sku_id'])    
    training_data  = data_total.copy()
    
    
    X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 5, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic',
        'eval_metric': 'auc', 'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.8}
#    param = {'max_depth': 5, 'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
#        'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic',
#        'eval_metric': 'auc'}
    num_round = 500
    param['nthread'] = 4
    param['eval_metric'] = "auc"
    plst = param.items()
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train( plst, dtrain, num_round, evallist, early_stopping_rounds=10)


    refer_date = '2016-04-08'
    dump_path = '../cache/trainset_%s.pkl' % refer_date
    data_test = pickle.load(open(dump_path, 'rb'))
    sub_label = data_test['label'].copy()
    sub_user_index = data_test[['user_id', 'sku_id']].copy()
    del(data_test['label'])
    del(data_test['user_id'])
    del(data_test['sku_id'])
    sub_trainning_data  = data_test.copy()

    sub_trainning_data.columns = training_data.columns
    test = xgb.DMatrix(sub_trainning_data)
    y = bst.predict(test)
    sub_user_index['user_id'] = [str(x) for x in sub_user_index['user_id']]
    sub_user_index['sku_id'] = [str(x) for x in sub_user_index['sku_id']]


    
    pred = sub_user_index.copy()
    #option 1 
    pred['label'] = (y>= 0.2)*1
    pred = pred.groupby('user_id').first().reset_index()
    pred_opt1 = pred.copy()
    print('opt1: %g' %pred_opt1['label'].sum())


    #option 2
#    pred = sub_user_index.copy()
#    pred['label'] = y
#    predunique = pred.groupby('user_id').first().reset_index().copy()
#    predunique = predunique.sort_values(by='label',ascending=False)
#    del pred['label']
#    predfinal = predunique.reset_index(drop=True).copy() 
#    predfinal['label'] = (predfinal['label']>=0.12)*1  
#    pred = pd.merge(pred, predfinal, how='left', on=['user_id', 'sku_id'])
#    pred = pred.fillna(0)
#    pred_opt2 = pred.copy()
#    print('opt2: %g' %pred_opt2['label'].sum())


    #real 
    y_true = sub_user_index.copy()
    y_true['label'] = sub_label
    y_true_all = pickle.load(open('../cache/purchase_'+refer_date+'.pkl', 'rb'))
    
    print('real: %g' %y_true['label'].sum())
    print('real_all: %g' %y_true_all['label'].sum())


    reportnew(pred_opt1, y_true)
    reportnew(pred_opt1, y_true_all)

#    reportnew(pred_opt2, y_true)



if __name__ == '__main__':
    xgboost_test()
#    xgboost_submission()
