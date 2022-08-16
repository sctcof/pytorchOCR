"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: get_key_label.py
@time: 2020/11/9 20:33

"""
from os import getcwd
wd = getcwd()
project_wd = wd[:wd.index('script')]

train_list_file = project_wd+'data/rec_data'+'/rec_train_list.txt'
test_list_file = project_wd+'data/rec_data'+'/rec_test_list.txt'
keys_file = project_wd+'data/rec_data'+'/key.txt'


fid_key = open(keys_file,'w+',encoding='utf-8')
keys = ''
with open(train_list_file,'r',encoding='utf-8') as fid_train:
    lines = fid_train.readlines()
    for line in lines:
        line = line.strip().split('\t')
        keys+=line[-1]

with open(test_list_file,'r',encoding='utf-8') as fid_test:
    lines = fid_test.readlines()
    for line in lines:
        line = line.strip().split('\t')
        keys+=line[-1]
#
# keys=keys.replace(' ','')
# keys_list=set(list(keys))
key = ''.join(list(set(list(keys))))
fid_key.write(key)