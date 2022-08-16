"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: test.py
@time: 2020/9/3 20:17

"""
import os
import argparse
from os import getcwd
def gen_train_file(args):
    label_path = args.label_path
    img_path = args.img_path
    files = os.listdir(img_path)
    with open(os.path.join(args.save_path,'rec_train_list.txt'),'w+',encoding='utf-8') as fid:
        for file in files:

            label_str = os.path.join(img_path,file)+'\t'+os.path.join(label_path,'gt_'+os.path.splitext(file)[0]+'.txt')+'\n'
            fid.write(label_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    wd = getcwd()
    project_wd = wd[:wd.index('script')]
    parser.add_argument('--label_path', nargs='?', type=str, default=project_wd+'data/ch4_training_gt')
    parser.add_argument('--img_path', nargs='?', type=str, default=project_wd+'data/ch4_training_images')
    parser.add_argument('--save_path', nargs='?', type=str, default=project_wd+'data/traindata')
    args = parser.parse_args()
    gen_train_file(args)