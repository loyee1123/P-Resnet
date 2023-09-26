import wandb
from wandb import init as wandb_initilizer
import tensorflow as tf
from tensorflow import one_hot
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras import models as KM
import numpy as np
from tensorflow.python.keras import metrics
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.utils import to_categorical
import math
from sklearn.metrics import confusion_matrix  # 混淆矩阵
import pandas as pd
import numpy as np
import os
import csv
import random
import matplotlib.pyplot as plt
from scipy import signal
import struct
import wave
from tensorflow import keras
from sklearn import metrics
from keras import optimizers, losses, activations, models, utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, Callback
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, normalization, add, LSTM, Bidirectional, concatenate, Activation, Bidirectional, TimeDistributed, \
    Lambda, multiply, Reshape
from keras_contrib.layers.normalization import instancenormalization
from keras import backend as K
# from sklearn.metrics import f1_score, accuracy_score, recall_score, specificity_score, balanced_accuracy_score
from sklearn.metrics import f1_score, accuracy_score, recall_score, balanced_accuracy_score
from imblearn.metrics import specificity_score
from keras.callbacks import TensorBoard
from sklearn.utils import class_weight
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve,auc,roc_auc_score

from tensorflow.python.keras import metrics




wandb_initilizer(entity='',project='')



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.45
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



def build_model(shape, match):
    assert len(shape) == 3

    model_input = KL.Input(shape=(shape[1], shape[2]))
    x = KL.Conv1D(filters=64,
                  kernel_size=5,
                  strides=1,
                  padding='same',
                  activation='relu',
                  kernel_initializer='glorot_uniform',
                  name='Conv1d_1' + '_' + match
                  )(model_input)

    x = KL.Conv1D(filters=32,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  activation='relu',
                  kernel_initializer='glorot_uniform',
                  name='Conv1d_2' + '_' + match
                  )(x)


    x = KL.Bidirectional(KL.LSTM(
        units=100,
        kernel_initializer='glorot_uniform',
        return_sequences=True,
        name='Bi_LSTM' + '_' + match))(x)
    x = KL.GlobalMaxPooling1D(name='Global_max_pooling' + '_' + match)(x)
    x = KL.Dropout(0.2)(x)
    hidden = KL.Dense(units=32,
                      activation='relu',
                      kernel_initializer='glorot_uniform',
                      name='dense_1' + '_' + match
                      )(x)
    hidden = KL.Dropout(0.1)(hidden)
    hidden = KL.Dense(units=16,
                      activation='relu',
                      kernel_initializer='glorot_uniform',
                      name='dense_2' + '_' + match
                      )(hidden)
    output = KL.Dense(units=2,
                      activation='softmax',
                      kernel_initializer='glorot_uniform',
                      name='output' + '_' + match
                      )(hidden)

    model = KM.Model(model_input, output)
    model.summary()

    return model

def _bn_relu(layer, dropout=0, **params):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D
    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        kernel_initializer=params["conv_init"])(layer)
    return layer


def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
                    layer,
                    params["conv_filter_length"],
                    params["conv_num_filters_start"],
                    subsample_length=subsample_length,
                    **params)
        layer = _bn_relu(layer, **params)
    return layer

def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    from keras.layers import Add
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                dropout=params["conv_dropout"] if i > 0 else 0,
                **params)
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length if i == 0 else 1,
            **params)
    layer = Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / params["conv_increase_channels_at"]) \
        * num_start_filters

def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            **params)
    layer = _bn_relu(layer, **params)
    return layer

def add_output_layer(layer, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    #layer = GlobalAveragePooling1D()(layer)
    layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    layer = GlobalAveragePooling1D()(layer)
    print('layer_shape=',layer.shape)
    return Activation('softmax')(layer)


def build_resnet34(**params):
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import ZeroPadding1D
    inputs = Input(shape=(10000, 1))
    #x = ZeroPadding1D(24)(inputs)
    print('input=', inputs.shape)
    if params.get('is_regular_conv', False):
        layer = add_conv_layers(inputs, **params)
    else:
        layer = add_resnet_layers(inputs, **params)
    output = add_output_layer(layer, **params)
    print('output=',output.shape)
    model = Model(inputs=[inputs], outputs=[output])
    # if params.get("compile", True):
    #     add_compile(model, **params)
    model.summary()
    return model

def to_num(array):
    return np.array([np.argmax(each) for each in array])


def model_test(test_x, model, one_hot=False):
    #print(test_x.shape)
    test_x = np.expand_dims(test_x, 2)
    predict_y = model.predict(test_x)

    if one_hot:
        return to_num(predict_y)
    return predict_y
def sen(Y_test, Y_pred, n):  # n为分类数

    sen = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    return sen


def pre(Y_test, Y_pred, n):
    pre = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:, i]) - tp
        pre1 = tp / (tp + fp)
        pre.append(pre1)

    return pre


def spe(Y_test, Y_pred, n):
    spe = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return spe


def ACC(Y_test, Y_pred, n):
    acc = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)

    return acc

class MeanTeacher:
    def __init__(self,
                 input_shape):
        assert len(input_shape) == 3
        self.predhis = []
        self.model = build_resnet34(conv_init=tf.keras.initializers.he_normal(seed=42),conv_filter_length=16, conv_num_filters_start=32, conv_subsample_lengths=[1, 2, 1, 2, 1, 2, 1, 2, 1, 5, 1, 5, 1, 5, 1, 1],
                           conv_activation='relu',conv_increase_channels_at=4,conv_num_skip=2, conv_dropout=0.2, num_categories=2, learning_rate=1e-3)
        self.teacher_model = build_resnet34(conv_init=tf.keras.initializers.he_normal(seed=42),conv_filter_length=16, conv_num_filters_start=32, conv_subsample_lengths=[1, 2, 1, 2, 1, 2, 1, 2, 1, 5, 1, 5, 1, 5, 1, 1],
                           conv_activation='relu',conv_increase_channels_at=4,conv_num_skip=2, conv_dropout=0.2, num_categories=2, learning_rate=1e-3)
        model_weights = self.model.get_weights()
        self.teacher_model.set_weights(model_weights)
        #self.model.summary()

        self.opt = tf.optimizers.Adam()
        self.loss_1 = CategoricalCrossentropy()
        self.loss_2 = MeanSquaredError()
        self.train_loss_1 = metrics.Mean(name='train_loss1')
        self.train_loss_2 = metrics.Mean(name='train_loss2')
        self.train_metric_1 = metrics.CategoricalAccuracy(name='train_accuracy1')
        self.train_metric_2 = metrics.MeanSquaredError(name='train_accuracy2')

        self.model.compile(optimizer=self.opt)  #loss和metric无意义s
        self.best_test_acc=0
        self.patience_count=0
        self.stop_patience_count=0

    def train(self, ori_label_data, label, un_label_data, test_p_x, test_n_x,aug=False, batch_size=32, end_epoch=100):

        delta = np.array([0.0, 0.0])

        print('ori labeled data shape:', ori_label_data.shape)
        print('unlabeled data shape:', un_label_data.shape)
        label_data=ori_label_data
        batch_size_label=96
        for epoch in range(1, end_epoch + 1):
            if epoch > 15:
                break

            acc_train = 0
            count_train = 0
            np.random.shuffle(un_label_data)
            total = list(zip(label_data, label))
            np.random.shuffle(total)
            label_data, label = zip(*total)
            label_data = np.array(label_data)
            label = np.array(label)
            print('epoch shuffle')

            start = 0
            start_l = 0

            k_thresh = self.get_thresh(un_label_data.shape[0], label_data.shape[0])
            weight_cl = self.cal_consistency_weight(epoch)

            weight_l = 1 - (epoch - 1) / end_epoch

            sharpen_weight = self.cal_sharpen_weight(epoch)

            while start+ batch_size < un_label_data.shape[0]:  #unlabelled data可以继续喂的时候

                input_ulx = np.expand_dims(un_label_data[start:start + batch_size], 2)
                teacher_pred = self.teacher_model(input_ulx + np.random.normal(0,0.01,input_ulx.shape), training=True).numpy() + delta
                teacher_pred = teacher_pred ** sharpen_weight / np.reshape(
                     np.sum(teacher_pred ** sharpen_weight, axis=-1), newshape=(-1, 1))  # 锐化teacher输出
                loss_2, model_pred = self.train_step(input_ulx, teacher_pred, self.loss_2, weight_cl)

                self.train_loss_2.update_state(loss_2)
                self.train_metric_2.update_state(teacher_pred, model_pred)


                if start_l + batch_size_label > k_thresh:
                    start_l = 0
                    total = list(zip(label_data, label))
                    np.random.shuffle(total)
                    label_data, label = zip(*total)
                    label_data = np.array(label_data)
                    label = np.array(label)
                if start_l + batch_size_label < label_data.shape[0]:
                    input_lx = np.expand_dims(label_data[start_l:start_l + batch_size_label], 2)
                    input_ly = label[start_l:start_l + batch_size_label]
                    loss_1, model_pred = self.train_step(input_lx, input_ly, self.loss_1, weight_l)


                    self.train_loss_1.update_state(loss_1)
                    self.train_metric_1.update_state(input_ly, model_pred)
                    acc_batch= float(self.train_metric_1.result())
                    count_train+=1
                    acc_train=(acc_train*(count_train-1)+acc_batch)/count_train

                logs = 'epoch: {} --- step: {}/{} --- loss1: {}, acc1: {}, loss2: {}, acc2: {}'
                print(logs.format(epoch, start + batch_size, un_label_data.shape[0],
                                   self.train_loss_1.result(),
                                   acc_train,
                                   self.train_loss_2.result(),
                                   self.train_metric_2.result()))

                self.train_loss_1.reset_states()
                self.train_loss_2.reset_states()
                self.train_metric_1.reset_states()
                self.train_metric_2.reset_states()


                start += batch_size
                start_l += batch_size_label

                self.update_ema(epoch)

            delta = self.cal_delta()
            print(f'delta={delta}')
            acc_test=self.test_on_epoch_end(test_p_x,test_n_x,epoch,acc_train)
            print(self.model.optimizer.lr)
            if self.reduce_lr_stopearly(acc_test):
                print('early stop')
                break


    def test_on_epoch_end(self,test_p_x,test_n_x,epoch,acc):

        case_test_pos = []
        case_test_neg = []
        test_p_y = [1 for _ in range(len(case_test_pos_count))]

        test_n_y = [0 for _ in range(len(case_test_neg_count))]

        test_y = test_p_y + test_n_y
        # test_y=np.expand_dims(test_y,1)
        test_p_x=np.expand_dims(test_p_x, 2)
        test_n_x=np.expand_dims(test_n_x, 2)

        pred_y_pos_segment = model_test(test_p_x,self.teacher_model)
        pred_y_neg_segment = model_test(test_n_x,self.teacher_model)
        loss_test=metrics.CategoricalAccuracy()


        pred_y_patient = []
        number_start = 0
        for i in case_test_pos_count:
            normal = (pred_y_pos_segment[number_start:number_start + i, 0].sum()) / i
            abnormal = (pred_y_pos_segment[number_start:number_start + i, 1].sum()) / i
            pred_y_patient.append([normal, abnormal])
            if normal > 1 - 0.20422535:
                case_test_pos.append(0)
            else:
                case_test_pos.append(1)
            number_start = number_start + i

        number_start = 0
        for i in case_test_neg_count:
            normal = (pred_y_neg_segment[number_start:number_start + i, 0].sum()) / i
            abnormal = (pred_y_neg_segment[number_start:number_start + i, 1].sum()) / i
            pred_y_patient.append([normal, abnormal])

            if normal > 1 - 0.20422535:
                case_test_neg.append(0)
            else:
                case_test_neg.append(1)
            number_start = number_start + i
        with open(save_name + f'{epoch:03d}_'+'predict_result.csv', "w", newline='', encoding='GBK') as f:
            writer = csv.writer(f, delimiter=',')
            for i in pred_y_patient:
                writer.writerow(i)


        correct_pos = 0
        correct_neg = 0
        for i in case_test_pos:
            if i == 1:
                correct_pos += 1
        for i in case_test_neg:
            if i == 0:
                correct_neg += 1
        print(correct_pos, correct_neg)
        pred_y = pred_y_patient

        pred_y_onehot = []
        for i in pred_y:
            if i[0] > 1 - 0.20422535:
                pred_y_onehot.append(0)
            else:
                pred_y_onehot.append(1)

        print(sen(test_y, pred_y_onehot, 1), pre(test_y, pred_y_onehot, 1), spe(test_y, pred_y_onehot, 1),
              ACC(test_y, pred_y_onehot, 1))


        TP_list = []
        TN_list = []
        pos_list = []
        neg_list = []
        acc_test = accuracy_score(test_y, pred_y_onehot)
        balance_acc = balanced_accuracy_score(test_y, pred_y_onehot)
        Sen = recall_score(test_y, pred_y_onehot)
        Spe = specificity_score(test_y, pred_y_onehot)

        test_y_array = np.hstack(test_y).reshape(-1, 1)
        pred_y_onehot_array = np.hstack(pred_y_onehot).reshape(-1, 1)
        pos = pred_y_onehot_array[test_y_array == 1]
        pos_list.append(pos.shape[0])
        neg = pred_y_onehot_array[test_y_array == 0]
        neg_list.append(neg.shape[0])
        TP_pred = pos[pos == 1]
        TP_list.append(TP_pred.shape[0])
        TN_pred = neg[neg == 0]
        TN_list.append(TN_pred.shape[0])

        pos_array = np.hstack(pos_list).reshape(-1, 1)
        neg_array = np.hstack(neg_list).reshape(-1, 1)
        TP_array = np.hstack(TP_list).reshape(-1, 1)
        TN_array = np.hstack(TN_list).reshape(-1, 1)
        acc_array = (TP_array + TN_array) / (pos_array + neg_array)
        Sen_array = TP_array / pos_array
        Spe_array = TN_array / neg_array
        Macc_array = (Sen_array + Spe_array) / 2
        FPR_array = 1 - Spe_array
        # Area = metrics.auc(FPR_array, Sen_array)

        pred_y_auc = []
        for i in pred_y:
            pred_y_auc.append(i[1])

        fpr, tpr, thresholds = roc_curve(test_y, pred_y_auc, pos_label=1)
        area = auc(fpr, tpr)
        print(
            'Accuracy score=%.4f,Balanced accuracy=%.4f, score,Sensitivity score=%.4f,Specificity score=%.4f,AUC score=%.4f' % (
                acc_test, balance_acc, Sen, Spe, area))
        index = np.array([acc_test, balance_acc, Sen, Spe, area]).reshape(1, 5)
        self.predhis.append(index)
        results = self.predhis

        with open(save_name + 'recording_result_index.csv', "w", newline='', encoding='GBK') as f:
            writer = csv.writer(f, delimiter=',')
            for i in results:  # 对于每一行的，将这一行的每个元素分别写在对应的列中
                writer.writerow(i)
        self.teacher_model.save(
            save_name + f'model_{epoch:03d}-{acc:.4f}.h5')

        wandb.log({
            'epoch': epoch,
            'lr':self.model.optimizer.learning_rate.numpy(),
            'test_acc': acc_test,
            'test_sen': Sen,
            'test_spe': Spe,
            'test_auc': area,
            'test_balanced_acc': balance_acc
        })
        return acc_test
    def reduce_lr_stopearly(self, test_acc, patience=3, stop_patience=15, factor=0.5):
        """
        Reduce learning rate if test accuracy does not improve after 'patience' epochs.
        Args:
            test_acc (float): Test accuracy at the current epoch.
            patience (int): Number of epochs to wait before reducing learning rate. Default: 6.
            factor (float): Factor by which the learning rate will be reduced. Default: 0.5.
        """
        # Retrieve the current learning rate
        lr = K.get_value(self.model.optimizer.lr)

        # If the test accuracy has not improved, increment the patience counter
        if test_acc <= self.best_test_acc:
            self.patience_count += 1
            self.stop_patience_count +=1
        # Otherwise, reset the patience counter and update the best test accuracy
        else:
            self.patience_count = 0
            self.stop_patience_count += 0
            self.best_test_acc = test_acc

        # If we have waited 'patience' epochs without improvement, reduce the learning rate
        if self.patience_count >= patience:
            new_lr = lr * factor
            K.set_value(self.model.optimizer.lr, new_lr)
            print(f'Learning rate reduced to {new_lr}.')

            # Reset the patience counter
            self.patience_count = 0

        if self.stop_patience_count >= stop_patience:
            return True
        else:
            return False

        # Print the current learning rate



    def cal_delta(self):
        pred = model_test(ssl_x, self.teacher_model, one_hot=False)

        pos_rate = np.asarray(pred[:, 1])
        pos_rate = pos_rate[pos_rate > 0.20422535]
        pos_rate = np.mean(pos_rate) if pos_rate.shape[0] > 0 else 0
        neg_rate = np.asarray(pred[:, 0])
        neg_rate = neg_rate[neg_rate > 1-0.20422535]
        neg_rate = np.mean(neg_rate) if neg_rate.shape[0] > 0 else 0

        if pos_rate > neg_rate:
            delta = np.array([min((pos_rate-neg_rate) * 2, 0.15), 0.0])
            print('pos rate is bigger')
        else:
            delta = np.array([0.0, min((neg_rate-pos_rate) * 2, 0.15)])
            print('neg rate is bigger')

        return delta

    def train_step(self, x, label,loss_func, weight):
        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            loss = weight * loss_func(label, pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, pred

    def update_ema(self, epoch):
        weights = self.model.get_weights()
        weights_t = self.teacher_model.get_weights()
        if epoch <= 10:
            alpha = 0.999
        else:
            alpha = 0.99
        for i in range(len(weights_t)):
            weights_t[i] = weights_t[i] * alpha + weights[i] * (1 - alpha)
        self.teacher_model.set_weights(weights_t)

    @staticmethod
    def data_augmentation(label_data, un_label_data):
        # 数据增强方法
        return label_data.copy()

    @staticmethod
    def get_thresh(un_s, l_s):
        th = un_s // 15
        if th < l_s:
            th = l_s
        if th > l_s * 3:
            th = l_s * 3
        return th

    @staticmethod
    def  cal_consistency_weight(epoch, init_ep=0.0, end_ep=100, init_w=0.0, end_w=5.0):
        """Sets the weights for the consistency loss"""
        if epoch > end_ep:
            weight_cl = end_w
        elif epoch < init_ep:
            weight_cl = init_w
        else:
            T = float(epoch - init_ep) / float(end_ep - init_ep)
            # weight_mse = T * (end_w - init_w) + init_w # linear
            weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w  # exp
        if weight_cl >= 1.0:
            weight_cl = 1.0
        return weight_cl

    @staticmethod
    def cal_sharpen_weight(epoch):
        if epoch < 10:
            return 2
        if epoch < 20:
            return 4
        return 5





if __name__ == '__main__':
    test_p_x = []
    test_n_x = []
    case_test_pos_count = []
    case_test_neg_count = []
    for i in []:
        X_test_path_pos = '' + str(i)
        X_test_path_neg = '' + str(i)
        name_test_p = os.listdir(X_test_path_pos)

        name_test_n = os.listdir(X_test_path_neg)

        name_test_p.sort(key=lambda x: (int(x.split('_')[2]), str(x.split('_')[3])))
        name_test_n.sort(key=lambda x: (int(x.split('_')[2]), str(x.split('_')[3])))
        for r in name_test_p:
            test_p_x.append(np.load(X_test_path_pos + '/' + r)[0])
        for r in name_test_n:
            test_n_x.append(np.load(X_test_path_neg + '/' + r)[0])
        name_pos_number_position = []
        name_neg_number_position = []
        for j in name_test_p:
            name_pos_number_position.append(j.split('_')[2] + '_' + j.split('_')[3])
        name_pos_number_position = list(set(name_pos_number_position))
        name_pos_number_position.sort(key=lambda x: (int(x.split('_')[0]), str(x.split('_')[1])))
        for j in name_test_n:
            name_neg_number_position.append(j.split('_')[2] + '_' + j.split('_')[3])
        name_neg_number_position = list(set(name_neg_number_position))
        name_neg_number_position.sort(key=lambda x: (int(x.split('_')[0]), str(x.split('_')[1])))


        case_test_pos_temp = []
        case_test_neg_temp = []

        for k in name_pos_number_position:
            pred_y = [0, 0]
            count_segment_all_type = 0
            for q in name_test_p:
                if k == q.split('_')[2] + '_' + q.split('_')[3]:
                    count_segment_all_type += 1

            case_test_pos_temp.append(count_segment_all_type)

        for k in name_neg_number_position:

            count_segment_all_type = 0
            for q in name_test_n:
                if k == q.split('_')[2] + '_' + q.split('_')[3]:
                    count_segment_all_type += 1

            case_test_neg_temp.append(count_segment_all_type)
        case_test_pos_count = case_test_pos_count + case_test_pos_temp
        case_test_neg_count = case_test_neg_count + case_test_neg_temp

    save_name = ""
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    # ssl_x_path = args.ssl_x_path  # 无标签数据
    single_p_x = []
    single_n_x = []
    # for i in range(2,10,1):
    for i in []:
        p_path = '' + str(i)
        n_path = '' + str(i)
        name_p = os.listdir(p_path)
        name_p.sort(key=lambda x: int((x.split('_')[2])))
        name_n = os.listdir(n_path)
        name_n.sort(key=lambda x: int((x.split('_')[2])))
        for z in name_p:
            single_p_x.append(np.load(p_path + '/' + z)[0])
        for z in name_n:
            single_n_x.append(np.load(n_path + '/' + z)[0])

    single_x = np.vstack((single_p_x, single_n_x))
    single_p_y = np.ones(shape=(len(single_p_x),))
    single_n_y = np.zeros(shape=(len(single_n_x),))
    single_y = np.hstack((single_p_y, single_n_y))
    file_path = save_name + 'model_{epoch:03d}-{accuracy:.4f}.h5'



    ssl_single_p_x=[]
    ssl_single_n_x=[]
    for i in []:
        ssl_p_path = '' + str(i)
        ssl_n_path = '' + str(i)
        ssl_name_p = os.listdir(ssl_p_path)
        ssl_name_p.sort(key=lambda x: int((x.split('_')[2])))
        ssl_name_n = os.listdir(ssl_n_path)
        ssl_name_n.sort(key=lambda x: int((x.split('_')[2])))
        for z in ssl_name_p:
            ssl_single_p_x.append(np.load(ssl_p_path + '/' + z)[0])
        for z in ssl_name_n:
            ssl_single_n_x.append(np.load(ssl_n_path + '/' + z)[0])
    ssl_x=np.vstack((ssl_single_p_x, ssl_single_n_x))
    tf.random.set_seed(3407)
    single_y = to_categorical(np.array(single_y), 2)
    interval = single_x.shape[1]
    mean_teacher = MeanTeacher(input_shape=(64, interval, 1))
    mean_teacher.train(single_x, single_y,ssl_x,test_p_x,test_n_x,)
