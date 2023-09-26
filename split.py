#split the dataset in patient level
import random
import os
import shutil
from random import shuffle


def split(full_list,shuffle=False,ratio1=0.1,ratio2=0.1):
    n_total = len(full_list)
    offset1 = int(n_total * ratio1)
    offset2 = int(n_total * (1-ratio2))
    if n_total==0 or offset1<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset1]
    sublist_2 = full_list[offset1:offset2]
    sublist_3 = full_list[offset2:]
    return sublist_1,sublist_2,sublist_3

def copyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + fname)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath + fname))

if __name__ == "__main__":

    source_filepath_pos = ''
    source_filepath_neg = ''
    name_pos= os.listdir(source_filepath_pos)
    name_neg=os.listdir(source_filepath_neg)
    des_pos=''
    des_neg=''
    shuffle(name_pos)
    shuffle(name_neg)

count_copy=0
count_folder=0
count_patient=0

name_pos_number=[]
name_neg_number=[]
for i in name_pos:
    name_pos_number.append(i.split('_')[0])

for i in name_neg:
    name_neg_number.append(i.split('_')[0])

name_pos_number = list(set(name_pos_number))
name_neg_number=list(set(name_neg_number))

for i in name_pos_number:
    for j in name_pos:
        if i==j.split('_')[0]:
            copyfile(source_filepath_pos + '/' + j, des_pos + '/'+str(count_folder)+'/')
            count_copy+=1
    count_patient += 1
    if count_patient >= 18:
        count_patient = 0
        count_folder += 1

count_copy=0
count_folder=0
count_patient=0
for i in name_neg_number:
    for j in name_neg:
        if i==j.split('_')[0]:
            copyfile(source_filepath_neg + '/' + j, des_neg + '/'+str(count_folder)+'/')
            count_copy+=1
    count_patient += 1
    if count_patient >= 70:
        count_patient = 0
        count_folder += 1
print(count_copy)





