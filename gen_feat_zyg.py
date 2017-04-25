# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 22:34:37 2017

@author: lenovo
"""

from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import numpy as np


action_1_path = "../data/JData_Action_201602.csv"
action_2_path = "../data/JData_Action_201603.csv"
action_3_path = "../data/JData_Action_201604.csv"
comment_path = "../data/JData_Comment.csv"
product_path = "../data/JData_Product.csv"
user_path = "../data/JData_User.csv"

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28", "2016-04-04", "2016-04-11", "2016-04-15"]

#refer_date_list = ["2016-04-01", "2016-04-02", "2016-04-03", "2016-04-04", "2016-04-05", 
#                   "2016-04-06", "2016-04-07", "2016-04-08", "2016-04-09", "2016-04-10" ]
refer_date_list = ["2016-04-07", "2016-04-08", "2016-04-09", "2016-04-10"]
##基本读取action操作
def get_actions_1():
    action = pd.read_csv(action_1_path)
    return action

def get_actions_2():
    action2 = pd.read_csv(action_2_path)
    return action2

def get_actions_3():
    action3 = pd.read_csv(action_3_path)
    return action3

def get_all_actions():
    """
    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    dump_path = '../cache/all_action.pkl' 
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        action_1 = get_actions_1()
        action_2 = get_actions_2()
        action_3 = get_actions_3()
        actions = pd.concat([action_1, action_2, action_3]) # type: pd.DataFrame
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions
    
def get_actions(end_date, delta_days):
    """

    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    dump_path = '../cache/all_action.pkl' 
    if not os.path.exists(dump_path):
        get_all_actions()
    else:
        actions = pickle.load(open('../cache/all_action.pkl', 'rb'))
        if type(end_date)==str:
            start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(delta_days)
        else:
            start_date = end_date - timedelta(delta_days)
        actions = actions[(pd.to_datetime(actions.time) >= start_date) & (pd.to_datetime(actions.time) < pd.to_datetime(end_date))]
    return actions
    
##增加user特征
def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1

def get_basic_user_feat():
    dump_path = '../cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path, 'rb'))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        user['age'] = user['age'].map(convert_age)
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        user = pd.concat([user['user_id'],user['user_reg_tm'], age_df, sex_df, user_lv_df], axis=1)
        pickle.dump(user, open(dump_path, 'wb'))
    return user

def get_userlife_user_feat(end_date):
    dump_path = '../cache/basic_user.pkl'
    user = pickle.load(open(dump_path, 'rb'))
    EndDate_train=pd.Timestamp(end_date)
    user['user_reg_tm'] = pd.to_datetime(user['user_reg_tm'])
    user=user[user['user_reg_tm']<=EndDate_train]
    user['userlife'] = [i for i in ( EndDate_train-user['user_reg_tm']).dt.days]
    return user
    

def get_accumulate_user_feat(end_date, delta_days):    
    feature = ['browse_num', 'addcart_num',	'delcart_num',	'buy_num',	'favor_num',	'click_num',
                  'buy_addcart_ratio',	'buy_browse_ratio','buy_delcart_ratio',	'buy_click_ratio',	'buy_favor_ratio'	]
    feature = [temp + '_' + str(delta_days)+'D' for temp in feature]            
    feature.append('user_id')     
    actions = get_actions(end_date, delta_days)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions['user_id'], df], axis=1)
    actions = actions.groupby(['user_id'], as_index=False).sum()
    actions['browse_num'+ '_'+str(delta_days)+'D'] = actions['action_1']
    actions['addcart_num'+ '_'+str(delta_days)+'D'] = actions['action_2']
    actions['delcart_num'+ '_'+str(delta_days)+'D'] = actions['action_3']
    actions['buy_num'+ '_'+str(delta_days)+'D'] = actions['action_4']
    actions['favor_num'+ '_'+str(delta_days)+'D'] = actions['action_5']
    actions['click_num'+ '_'+str(delta_days)+'D'] = actions['action_6']
    actions['buy_addcart_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_1']
    actions['buy_browse_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_2']
    actions['buy_delcart_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_3']
    actions['buy_click_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_5']
    actions['buy_favor_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_6']
    actions = actions[feature]
    actions = actions.fillna(0)
    actions[np.isinf(actions)] = 0
    return actions


##增加product特征
def get_basic_product_feat():
    dump_path = '../cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path, 'rb'))
    else:
        product = pd.read_csv(product_path)
        attr1_df = pd.get_dummies(product["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product["a3"], prefix="a3")
        product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
        pickle.dump(product, open(dump_path, 'wb'))
    return product
    

def get_accumulate_product_feat(end_date, delta_days):        
    feature = ['browse_num', 'addcart_num',	'delcart_num',	'buy_num',	'favor_num',	'click_num',
                  'buy_addcart_ratio',	'buy_browse_ratio','buy_delcart_ratio',	'buy_click_ratio',	'buy_favor_ratio'	]
    feature = [temp + '_' + str(delta_days)+'D' for temp in feature]            
    feature.append('sku_id')     
    actions = get_actions(end_date, delta_days)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions['sku_id'], df], axis=1)
    actions = actions.groupby(['sku_id'], as_index=False).sum()
    actions['browse_num'+ '_'+str(delta_days)+'D'] = actions['action_1']
    actions['addcart_num'+ '_'+str(delta_days)+'D'] = actions['action_2']
    actions['delcart_num'+ '_'+str(delta_days)+'D'] = actions['action_3']
    actions['buy_num'+ '_'+str(delta_days)+'D'] = actions['action_4']
    actions['favor_num'+ '_'+str(delta_days)+'D'] = actions['action_5']
    actions['click_num'+ '_'+str(delta_days)+'D'] = actions['action_6']
    actions['buy_addcart_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_1']
    actions['buy_browse_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_2']
    actions['buy_delcart_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_3']
    actions['buy_click_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_5']
    actions['buy_favor_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_6']
    actions = actions[feature]
    actions = actions.fillna(0)
    actions[np.isinf(actions)] = 0
    return actions 

def get_comments_product_feat(end_date, delta_days):
    feature = ['comment_num', 'has_bad_comment', 'bad_comment_rate']
    feature = [temp + '_' + str(delta_days)+'D' for temp in feature] 
    feature.append('sku_id')  
    dump_path = '../cache/JData_Comment.pkl'
    if os.path.exists(dump_path):
        comments = pickle.load(open(dump_path, 'rb'))
    else:
        comments = pd.read_csv(comment_path)
        pickle.dump(comments, open(dump_path, 'wb'))

    start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(delta_days)
    comment_date_refer = comment_date[0]
    for date in reversed(comment_date):
        if datetime.strptime(date, '%Y-%m-%d') < start_date:
            comment_date_refer = date
            break
    comments = comments[(comments.dt == comment_date_refer)]
    comments['comment_num'+'_'+str(delta_days)+'D'] = comments['comment_num']
    comments['has_bad_comment'+'_'+str(delta_days)+'D'] = comments['has_bad_comment']
    comments['bad_comment_rate'+'_'+str(delta_days)+'D'] = comments['bad_comment_rate']
    comments = comments[feature]                        
    return comments
    
##基本action累积操作
def get_accumulate_action_feat(end_date, delta_days):
    feature = ['browse_num', 'addcart_num',	'delcart_num',	'buy_num',	'favor_num',	'click_num',
                  'buy_addcart_ratio',	'buy_browse_ratio','buy_delcart_ratio',	'buy_click_ratio',	'buy_favor_ratio'	]
    feature = [temp + '_' + str(delta_days)+'D' for temp in feature] 
    feature.append('user_id')     
    feature.append('sku_id')     
    actions = get_actions(end_date, delta_days)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions[['user_id','sku_id']], df], axis=1)
    actions = actions.groupby(['user_id','sku_id'], as_index=False).sum()
    actions['browse_num'+ '_'+str(delta_days)+'D'] = actions['action_1']
    actions['addcart_num'+ '_'+str(delta_days)+'D'] = actions['action_2']
    actions['delcart_num'+ '_'+str(delta_days)+'D'] = actions['action_3']
    actions['buy_num'+ '_'+str(delta_days)+'D'] = actions['action_4']
    actions['favor_num'+ '_'+str(delta_days)+'D'] = actions['action_5']
    actions['click_num'+ '_'+str(delta_days)+'D'] = actions['action_6']
    actions['buy_addcart_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_1']
    actions['buy_browse_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_2']
    actions['buy_delcart_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_3']
    actions['buy_click_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_5']
    actions['buy_favor_ratio'+ '_'+str(delta_days)+'D'] = actions['action_4'] / actions['action_6']
    actions = actions[feature]
    actions = actions.fillna(0)
    actions[np.isinf(actions)] = 0
    return actions

def make_feat(refer_date):
    ##get basic action
    actions = get_actions(refer_date, 30)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions[['user_id','sku_id']], df], axis=1)
    del df
    actions = actions.groupby(['user_id','sku_id'], as_index=False).sum()
    actions = actions[(actions['action_1']>0) | (actions['action_6']>0)]
    actions['user_id'] = actions['user_id'].astype(int)
    actions = actions.fillna(0)
    ##get basic user
    users = get_basic_user_feat()
    users_life = get_userlife_user_feat(refer_date)
    users_life = users_life[['user_id','userlife']]
    users = pd.merge(users, users_life, how='left', on='user_id')
    del users['user_reg_tm']
    users_3D = get_accumulate_user_feat(refer_date, 3)
    users_3D['user_id'] = users_3D['user_id'].astype(int)
    users_7D = get_accumulate_user_feat(refer_date, 7)
    users_7D['user_id'] = users_7D['user_id'].astype(int)
    users_15D = get_accumulate_user_feat(refer_date, 15)
    users_15D['user_id'] = users_15D['user_id'].astype(int)
    users_30D = get_accumulate_user_feat(refer_date, 30)
    users_30D['user_id'] = users_30D['user_id'].astype(int)
    users = pd.merge(users, users_3D, how='left', on='user_id')
    users = pd.merge(users, users_7D, how='left', on='user_id')
    users = pd.merge(users, users_15D, how='left', on='user_id')
    users = pd.merge(users, users_30D, how='left', on='user_id')
    users_keep = users[(users['click_num_30D']>0) | (users['browse_num_30D']>0)].copy()
    ##get basic product
    product = get_basic_product_feat()
    comment_3D = get_comments_product_feat(refer_date, 3)
    product_3D =get_accumulate_product_feat(refer_date, 3)
    comment_7D = get_comments_product_feat(refer_date, 7)
    product_7D =get_accumulate_product_feat(refer_date, 7)    
    comment_15D = get_comments_product_feat(refer_date, 15)
    product_15D =get_accumulate_product_feat(refer_date, 15)    
    comment_30D = get_comments_product_feat(refer_date, 30)
    product_30D =get_accumulate_product_feat(refer_date, 30)  
    product = pd.merge(product, comment_3D, how='left', on='sku_id')
    product = pd.merge(product, product_3D, how='left', on='sku_id')
    product = pd.merge(product, comment_7D, how='left', on='sku_id')
    product = pd.merge(product, product_7D, how='left', on='sku_id')
    product = pd.merge(product, comment_15D, how='left', on='sku_id')
    product = pd.merge(product, product_15D, how='left', on='sku_id')
    product = pd.merge(product, comment_30D, how='left', on='sku_id')
    product = pd.merge(product, product_30D, how='left', on='sku_id')
    product = product.fillna(0)
    product_keep = product[(product['click_num_30D']>0) | (product['browse_num_30D']>0)].copy()
    ###
    actions = pd.merge(actions, product_keep, how='left', on='sku_id')
    actions_keep = actions.dropna(axis = 0).copy()
    actions_keep = pd.merge(actions_keep, users_keep, how='left', on='user_id')
    actions_keep = actions_keep.dropna()
    return actions_keep

                      
    
def make_label(refer_date):
    end_date = datetime.strptime(refer_date, '%Y-%m-%d') + timedelta(days=5)
    actions = get_actions(end_date, 5)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions[['user_id','sku_id']], df], axis=1)
    actions = actions.groupby(['user_id','sku_id'], as_index=False).sum()
    actions = actions[actions['action_4']>0].reset_index(drop = True)
    actions['label'] = np.sign(actions['action_4']) 
    label = actions[['user_id','sku_id','label']]
    label['user_id'] = label['user_id'].astype(int)
    return label
    
def make_train(refer_date):
    actions_keep = make_feat(refer_date)
    label = make_label(refer_date)
    actions_keep = pd.merge(actions_keep, label, how='left', on=['user_id','sku_id'])
    actions_keep = actions_keep.fillna(0)
    return actions_keep
        
def make_test(refer_date):
    actions_keep = make_feat(refer_date)
    return actions_keep

def make_train_list(refer_date_list):
    for refer_date in refer_date_list:
        print('Making Train Set '+refer_date)
        dump_path = '../cache/trainset_%s.pkl' % refer_date
        if not os.path.exists(dump_path):
            data = make_train(refer_date)
            pickle.dump(data, open(dump_path, 'wb'))
   
def reportnew(pred, y_true):
    pred = pred[['user_id','sku_id','label']]
    y_true = y_true[['user_id','sku_id','label']]

    actions = y_true[y_true['label']==1]
    result = pred[pred['label']==1]

    # 实际用户商品
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    all_user_real_set = actions['user_id'].unique()

    # 预测用户商品
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)
    all_user_test_set = result['user_id'].unique()


    pos = 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
    all_item_acc = 1.0 * pos / len(all_user_test_item_pair)
    all_item_recall = 1.0 * pos / len(all_user_real_set)
    print ('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print ('所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_item_acc * all_item_recall / (5.0 * all_item_recall + all_item_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    
    score = 0.4 * F11 + 0.6 * F12
    print ('F11=' + str(F11))
    print ('F12=' + str(F12))
    print ('score=' + str(score))
            
if __name__ == '__main__':
    for refer_date in refer_date_list:
        begin = datetime.now()
        print('Making Train Set '+refer_date)
        dump_path = '../cache/trainset_%s.pkl' % refer_date
        data = make_train(refer_date)
        pickle.dump(data, open(dump_path, 'wb'))
        del data
        end = datetime.now()
        print(end-begin)

    begin = datetime.now()
    refer_date = "2016-04-15"
    dump_path = '../cache/trainset_%s.pkl' % refer_date
    print('Making Test Set '+refer_date)
    dump_path = '../cache/testset_%s.pkl' % refer_date
    data = make_test(refer_date)
    pickle.dump(data, open(dump_path, 'wb'))
    del data
    end = datetime.now()
    print(end-begin)

    for refer_date in refer_date_list:
        begin = datetime.now()
        print('Making Purchase Set '+refer_date)
        dump_path = '../cache/purchase_%s.pkl' % refer_date
        data = make_label(refer_date)
        pickle.dump(data, open(dump_path, 'wb'))
        del data
        end = datetime.now()
        print(end-begin)
    
        
#import pickle
#
#actions = pickle.load(open('../cache/all_action.pkl', 'rb'))
#product = pickle.load(open('../cache/basic_product.pkl', 'rb'))
#user = pickle.load(open('../cache/basic_user.pkl', 'rb'))
#comment = pickle.load(open('../cache/JData_Comment.pkl', 'rb'))
#    
#temp1 = pickle.load(open('../cache/trainset_2016-04-01.pkl', 'rb'))
#temp2 = pickle.load(open('../cache/testset_2016-04-15.pkl', 'rb'))


