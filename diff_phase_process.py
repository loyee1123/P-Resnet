
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy import signal
import struct
import os
import csv
import random
import pandas as pd
from PIL import Image
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings



def read_wav_data(filename):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    print('filename=', filename)
    wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes()  # 获取帧数
    # print('num_frame=',num_frame)
    num_channel = wav.getnchannels()  # 获取声道数
    # print('num_channel=',num_channel)
    framerate = wav.getframerate()  # 获取帧速率
    # print('framerate=',framerate)
    num_sample_width = wav.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame)  # 读取全部的帧
    wav.close()  # 关闭流
    # wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
    wave_data = np.fromstring(str_data, dtype=np.int16)
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))  # wave幅值归一化
    # wave_data = (wave_data-np.mean(wave_data))/(np.std(wave_data))
    # print(wave_data.shape)
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T  # 将矩阵转置

    return wave_data, framerate, num_sample_width, num_channel


def band_pass_filter(original_signal, order, fc1, fc2, fs):
    '''
    中值滤波器
    :param original_signal: 音频数据
    :param order: 滤波器阶数
    :param fc1: 截止频率
    :param fc2: 截止频率
    :param fs: 音频采样率
    :return: 滤波后的音频数据
    '''
    b, a = signal.butter(N=order, Wn=[2 * fc1 / fs, 2 * fc2 / fs], btype='bandpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal


def enframe(signal, nw, inc, winfunc):
    '''
    将音频信号转化为帧。

    参数含义：

    signal:原始音频型号

    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)

    inc:相邻帧的间隔（同上定义）

    '''
    '''
    #不足用零补齐
    signal_length=len(signal) #信号总长度
    print('signal_length=',signal_length)

    if signal_length<=nw: #若信号长度小于一个帧的长度，则帧数定义为1

        nf=1

    else: #否则，计算帧的总长度

        nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))
        #nf=int(np.floor((1.0*signal_length-nw+inc)/inc))
        pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度

        zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作

        pad_signal=np.concatenate((signal,zeros)) #填补后的信号记为pad_signal

        indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵

        indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵

        frames=pad_signal[indices] #得到帧信号

        win=np.tile(winfunc,(nf,1)) #window窗函数，这里默认取1

        return frames*win  #返回帧信号矩阵
    '''

    # 不足部分舍弃
    signal_length = len(signal)  # 信号总长度
    print('signal_length=', signal_length)

    if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1

        nf = 1

    else:  # 否则，计算帧的总长度

        nf = int(np.floor((1.0 * signal_length - nw + inc) / inc))
        pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
        indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                               (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
        # print('indices=',indices)
        indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
        frames = signal[indices]  # 得到帧信号
        win = np.tile(winfunc, (nf, 1))  # window窗函数，这里默认取1
        # return frames*win  #返回帧信号矩阵
        return frames


def wav_show(wave_data, fs, name_file):  # 显示出来声音波形
    time = np.arange(0, len(wave_data)) * (1.0 / fs)  # 计算声音的播放时间，单位为秒
    # 画声音波形
    plt.plot(time, wave_data)
    plt.savefig('./dataset/source_data/train_val_set/val_set_bmp/' + name_file + '.png')
    plt.close()
    # plt.show()


if (__name__ == '__main__'):
    for q in range(0,10,1):
        neg_patch_number = 0
        pos_patch_number = 0
        filepath_positive='/home/luoyi/PCG/semi_remote/data/10_folder_patientslevel_recordingdata/neg_recording/'+str(q)
        name_positive= os.listdir(filepath_positive)  # 得到文件夹下的所有文件名称
        #name_positive.sort(key=lambda x: int((x.split('_')[0])))
        heart_sound_list = []
        lable_list = []
        heart_sound_path_list = []
        lable_path_list = []
        for i in range(0, len(name_positive), 1):
            wave_data, fs, num_sample_width, num_channel = read_wav_data(filepath_positive+ '/' + name_positive[i])
            name_file = name_positive[i].split('.')[0]
            wave_data_after_filtering = band_pass_filter(wave_data, 5, 25, 400, fs)  # (1,num_frame)五阶巴特沃夫滤波器滤波
            lable = np.array([1])
            lable_num = 1
            winfunc = signal.hamming(10000)  # 帧长 5s*2000Hz
            end = len(wave_data_after_filtering[0])
            for k in range(0, 1, 1):
                wave_data_phase = wave_data_after_filtering[0, 200 * k:end]
                # print('wave_data_phase=',wave_data_phase.shape)
                if len(wave_data_phase) < 10000:
                    break
                Frame = enframe(wave_data_phase, 10000, 5000, winfunc)
                pos_patch_number = pos_patch_number + Frame.shape[0]
                num = Frame.shape[0]
                name_file = name_positive[i].split('.')[0]

                for j in range(0, num, 1):
                    heart_sound_save_path = '/home/luoyi/PCG/semi_remote/data/10_folder_patientslevel_recordingdata/neg_segment/'+str(q)+ '/' + 'heart_sound_' + name_file + '_' + str(k) + '_' + str(j) + '.npy'
                    heart_sound = Frame[j, :]
                    heart_sound = heart_sound.reshape(1, -1)
                    heart_sound_list.append(heart_sound)
                    np.save(heart_sound_save_path, heart_sound)
                    heart_sound_path_list.append(heart_sound_save_path)
                        #lable_save_path = '/home/luoyi/PCG/data_2022/murmur/diff_phase/all/' + str(
                         #   m) + '/' + 'label_' + name_file + '_' + str(k) + '_' + str(j) + '.npy'
                        #np.save(lable_save_path, lable)
                        #lable_list.append(lable)
                        #lable_path_list.append(lable_save_path)

            heart_sound_save_name_train = '/home/luoyi/PCG/semi_remote/data/10_folder_patientslevel_recordingdata/neg_segment/'+str(q) + '/' + 'neg_heart_sound_path_.csv'
            #lable_save_name_train = '/home/luoyi/PCG/data_2022/murmur/diff_phase/all/' + str(m) + '_label_path.csv'

            heart_sound_path_csv_train = open(heart_sound_save_name_train, 'w')
            for index in range(len(heart_sound_path_list)):
                heart_sound_path_csv_train.writelines(heart_sound_path_list[index] + '\n')
            #lable_path_csv_train = open(lable_save_name_train, 'w')
            #for index in range(len(lable_path_list)):
             #   lable_path_csv_train.writelines(lable_path_list[index] + '\n')

        print('pos_patch_number=', pos_patch_number)
        print('neg_patch_number=', neg_patch_number)
