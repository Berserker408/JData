# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 21:12:54 2017

@author: lenovo
"""

import pandas as pd
import datetime
import pickle
from gen_feat_zyg import reportnew

data = pd.read_csv('../data/user_product_table.csv')

datanew= data[(data['browse_num']>0)].copy()
datanew= data[(data['click_num']>0)].copy()

datanew= data[(data['browse_num_30D']>0) | (data['click_num_30D']>0)].copy()

pickle.dump(data, open('../data/user_product_table.pkl', 'wb'))

begin = datetime.datetime.now()
data = pd.read_csv('../data/user_product_table.csv')
end = datetime.datetime.now()
print(end-begin)

begin = datetime.datetime.now()
data = pickle.load(open('../data/user_product_table.pkl', 'rb'))
end = datetime.datetime.now()
print(end-begin)

refer_date_list = ["2016-04-01", "2016-04-02", "2016-04-03", "2016-04-04", "2016-04-05", 
                   "2016-04-06", "2016-04-07", "2016-04-08", "2016-04-09", "2016-04-10" ]
for date in refer_date_list:                      
    print(date)
    temp = pickle.load(open('../cache/trainset_'+date+'.pkl', 'rb'))
    print(temp['label'].sum())
    
for date in refer_date_list:
    print(date)
    trainset = pickle.load(open('../cache/trainset_'+date+'.pkl', 'rb'))
    purchase = pickle.load(open('../cache/purchase_'+date+'.pkl', 'rb'))
    trainset = trainset[['user_id','sku_id','label']]
    reportnew(trainset, purchase)
    
for date in refer_date_list:
    print(date)
    purchase = pickle.load(open('../cache/purchase_'+date+'.pkl', 'rb'))
    
    