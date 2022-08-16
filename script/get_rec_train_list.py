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
def gen_rec_train_file(args):
    img_gt_path = args.img_gt_path
    save_path = args.save_path
    print(save_path)
    with open(os.path.join(save_path, 'rec_train_list.txt'), 'w+', encoding='utf-8') as fid:
        with open(img_gt_path) as f:
            lines=f.readlines()
            for line in lines:
                img_info = line.strip().split(',')
                img_file_name = img_info[0]
                img_file_text = img_info[1].strip()[1:-1]
                rec_img_info_str = save_path + '/train_img/'+img_file_name+'\t'+img_file_text+'\n'
                fid.write(rec_img_info_str)

def gen_rec_test_file(args):
    img_test_gt_path = args.img_test_gt_path
    save_path = args.save_path
    print(save_path)
    with open(os.path.join(save_path, 'rec_test_list.txt'), 'w+', encoding='utf-8') as fid:
        with open(img_test_gt_path) as f:
            lines = f.readlines()
            for line in lines:
                img_info = line.strip().split(',')
                img_file_name = img_info[0]
                img_file_text = img_info[1].strip()[1:-1]
                rec_img_info_str = save_path + '/test_img/' + img_file_name + '\t' + img_file_text + '\n'
                fid.write(rec_img_info_str)
    # with open()
    # files = os.listdir(img_path)
    #
    # with open(os.path.join(args.save_path,'rec_train_list.txt'),'w+',encoding='utf-8') as fid:
    #     for file in files:
    #
    #         label_str = os.path.join(img_path,file)+'\t'+os.path.join(label_path,'gt_'+os.path.splitext(file)[0]+'.txt')+'\n'
    #         fid.write(label_str)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    wd = getcwd()
    project_wd = wd[:wd.index('script')]
    # parser.add_argument('--label_path', nargs='?', type=str, default=project_wd+'data/ch4_training_gt')
    parser.add_argument('--img_gt_path', nargs='?', type=str, default=project_wd+'data/rec_data/train_img/gt.txt')
    parser.add_argument('--save_path', nargs='?', type=str, default=project_wd+'data/rec_data')
    parser.add_argument('--img_test_gt_path', nargs='?', type=str, default=project_wd+'data/rec_data/test_img/gt.txt')
    args = parser.parse_args()
    gen_rec_train_file(args)
    gen_rec_test_file(args)