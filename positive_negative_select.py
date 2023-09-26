import os
import csv
import shutil
def copyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + fname)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath + fname))

source_filepath=''
des_filepath_pos=''
des_filepath_neg=''
name=os.listdir(source_filepath)
name.sort(key=lambda x:int((x.split('_')[0])))
label_all=[]
source_neg=[]
source_pos=[]
csv_file=csv.reader(open('','r'))
for line in csv_file:
    label_all.append(line[0])
for i in range(0,len(label_all),1):
    if int(label_all[i])==0: #正常的
        source_neg.append(name[i])
    if int(label_all[i])==1: #阳性
        source_pos.append(name[i])
for i in source_pos:
    copyfile(source_filepath+'/'+i,des_filepath_pos+'/')
for i in source_neg:
    copyfile(source_filepath+'/'+i,des_filepath_neg+'/')





