# -*- coding: utf-8 -*-
# @Time    : 2022/7/8 0:21
# @Author  : Nevaeh
# @File    : standard_EIRS_multiclass.py
# @Software: PyCharm

import numpy as np

import pandas as pd
import heapq
from scipy.spatial.distance import pdist, squareform
# from scipy import stats
from scipy.stats import mode
from numpy import *
from sklearn.model_selection import KFold,StratifiedKFold
import copy

import warnings

warnings.filterwarnings("ignore")

def create_dist_array(array):
    """use points create array"""
    return squareform(pdist(array, metric='euclidean'))

def create_norm_array(array):  # 归一化矩阵
    norm = array / (np.mean(array, axis=0))  # 归一化平均值
    length = len(norm)
    return np.round(norm * (np.round(length / np.mean(norm, axis=0), 4)), 1)  # 和排名保持一致

def create_rank_array(array):  # 排序矩阵
    # todo 以前
    # rank = []
    # n = len(array[0, :])
    # for i in range(len(array)):
    #     rank_one = np.zeros(n)
    #     index = sorted(range(n), key=lambda k: array[i, k])
    #     for j in range(len(index)):
    #         rank_one[index[j]] = j
    #     rank.append(list(rank_one))
    #
    # return np.array(rank)
    # todo 以前
    rank = []
    n = len(array[0, :])
    for i in range(n):
        obj = pd.Series(array[i])
        b = obj.rank(method='average')
        rank.append(b)
    return np.array(rank)

class BtmkHeap(object):
    def __init__(self, k):
        self.k = k
        #空堆
        self.data = []

    def Push(self, elem):
        # Reverse elem to convert to max-heap
        elem = -elem
        # Using heap algorighem
        #如果前k小的值没找全，就继续找
        if len(self.data) < self.k:
            #heappush为heap增加元素：为空堆中加elem
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                #heapreplace删除堆顶后将新元素放到堆顶，然后下沉,删除最大值并添加新值
                heapq.heapreplace(self.data, elem)

    def BtmK(self):
        return sorted([-x for x in self.data])

def RD_fuction_MNN_7(train):
    "优化检测的噪声 取余 分层翻转 "
    K = [3, 5, 7, 9, 11] # [3, 5, 7, 9, 11]
    T = 0
    train = train.values
    m,n = train.shape
    suspected_noise = []
    suspected_noise_dic = {}
    last = []
    break_flag = False
    flag_5 = True
    flag_4 = True
    for i in range(m):
        suspected_noise_dic[i] = 0
       #create_dist_array计算距离矩阵
    array_dist = create_dist_array(train[:, 1:])  # 全部数据的距离矩阵
    array_predict = array_dist  # 每次判定一个测试集点的类别
    # array_norm = create_norm_array(array_predict)
    array_norm = create_rank_array(array_predict)
    array_norm = array_norm + array_norm.T


    while True:
        T = T + 1
        if T > 20:
            break
        dic = {}
        #pred_label_array: 所有样本5个k值的投票结果
        # [[1,0,1,0,1],[1,2,0,0,0]]
        pred_label_array = np.zeros((0,len(K)))
        #单个样本5个k值的投票结果
        #{3:1, 5:0, 7:0, 9:1, 11:2}
        pred_labels = {}
        for i in range(m):
            dic[i] = 0
            # dic_pred_label[i] = 0
        for i in range(m):
            k_label = []  # 存储每个测试集的的k近邻的值
            index_k_list = []
            th = BtmkHeap(K[-1] + 1)  # 排序前k小的值
            for u in list(array_norm[:m, i]):  # 最后一行为 0 所以不是m+1
                th.Push(u)
            k_list = th.BtmK()[1:]
            for j in range(K[-1]):
                index_k = list(array_norm[:m, i]).index(k_list[j])
                index_k_list.append(index_k)
                k_label.append(train[index_k, 0])
            # predict_label=statistics.mode(k_label)
            for o in K:
                predict_label = (max(k_label[:o], key=k_label[:o].count))  # 出现次数最多的标签
                pred_labels[o]=predict_label
                if predict_label != train[i, 0]:
                    dic[i] += 1
            pred_label_array=np.r_[pred_label_array,np.array(list(pred_labels.values())).reshape((1,5))]
            #axis=1,对列取mode，统计5次投票的预测结果(96,1)
            predict_res=mode(pred_label_array,axis=1)[0]
        temp_5 = []
        temp_4 = []
        temp_3 = []

        for w in dic.keys():
            if dic[w] == 5 and flag_5:
                temp_5.append(w)
            elif dic[w] >= 4 and flag_4:
                temp_4.append(w)
                flag_5 = False
            elif dic[w] >= 3:
                flag_5 = False
                flag_4 = False
                temp_3.append(w)
        if len(temp_5) != 0:
            print("Five")
            temp = temp_5
        elif len(temp_4) != 0:
            print("four")
            temp = temp_4
        elif len(temp_3) != 0:
            print("Three")
            temp = temp_3
        else:
            break
        noise_u=np.zeros((0,n))
        flip_u=np.zeros((0,n))
        for u in temp:
            noise_u=np.r_[noise_u, train[u,:].reshape((1,n))]
            suspected_noise_dic[u] += 1
            train[u,0]=predict_res[u]
            flip_u=np.r_[flip_u, train[u,:].reshape((1,n))]
            # if train[u, 0] == 1:
            #     train[u, 0] = 0  # todo  画图改成了0
            # else:
            #     train[u, 0] = 1
        print("length", len(temp))

        if len(temp) == len(last):  # 比较两次是否一样的 一样的就停止下来
            sorted(temp)
            sorted(last)
            for s in range(len(last)):
                if temp[s] != last[s]:
                    break
                if s == len(last) - 1 and temp[s] == last[s]:
                    break_flag = True
        if break_flag or len(temp) == 0:
            break

        last = copy.deepcopy(temp)

    for q in dic.keys():
        if suspected_noise_dic[q] % 2 == 1:
        #多分类中没有反复翻转，或者概率较小，改成=3、4、5的错误分类，都作为怀疑对象进行输出
        # if suspected_noise_dic[q]>=1:
            suspected_noise.append(q)

    return suspected_noise,predict_res

from scipy.spatial import cKDTree

def find_general_nn(data1, data2, k1, k2, n_jobs=2):
    k_index_1 = cKDTree(data1).query(x=data2, k=k1, n_jobs=n_jobs)[1]
    # k_index_2 = cKDTree(data2).query(x=data1, k=k2, n_jobs=n_jobs)[1]
    k_index_2=copy.deepcopy(k_index_1)

    mutual_1 = {}
    # mutual_2 = {}
    gnn_1={}

    for index_2 in range(data2.shape[0]):
        for index_1 in range(data1.shape[0]):
            if index_2 in k_index_2[index_1]:
                mutual_1.setdefault(index_2,[]).append(index_1)
            # else if :
            # '''防止样本点纵向无近邻，字典取值出错，现将其自身序值加入mutual_1,后续会差集被剔除'''
            #     mutual_1.setdefault(index_2, []).append(index_2)
                # mutual_2.setdefault(index_1,[]).append(index_2)
                # mutual_1.append(index_1)
                # mutual_2.append(index_2)
    for index_2 in range(data2.shape[0]):
        nn_row_index_2 = k_index_1[index_2]
        try:
            nn_column_index_2 = mutual_1[index_2]
        except:
            nn_column_index_2 = [index_2]
        gnn_1.setdefault(index_2,(set(list(nn_row_index_2))|set(nn_column_index_2)).difference(set([index_2])))
    return gnn_1

def GNN_1(train):
    K = [3, 5, 7, 9, 11]  #
    # K=[3]
    T = 0
    train = train.values
    train_x=train[:,1:]
    train_y=train[:,0]
    m,n = train.shape
    suspected_noise = []
    suspected_noise_dic = {}
    last = []
    break_flag = False
    flag_5 = True
    flag_4 = True
    for i in range(m):
        suspected_noise_dic[i] = 0

    gnn_ks = {}
    for k in K:
        gnn = find_general_nn(train_x, train_x, k1=k + 1, k2=k + 1)
        gnn_ks[k] = gnn

    while True:
        T = T + 1
        if T > 20:
            break
        #记录5个k值的预测结果scipy的mode函数,m*k
        pred_label_array = np.zeros((m,0))

        vote={}
        for i in range(m):
            vote[i]=0
        for k in K:
            gnn = gnn_ks[k]
            pred_nn_labels = []  # 记录单个k对所有样本的预测结果
            for i in range(m):
                gnn_idx=list(gnn[i])
                pred_nn_label=mode(train_y[gnn_idx],axis=0)[0][0]
                pred_nn_labels.append(pred_nn_label)
                if pred_nn_label != train_y[i]:
                    vote[i]+=1
                # print(pred_nn_labels)
            #pred_label_array: 所有样本5个k值的投票结果
            pred_label_array=np.c_[pred_label_array,pred_nn_labels]
            # axis=1,对列取mode，统计5次投票的预测结果(96,1)
            predict_res = mode(pred_label_array, axis=1)[0]

        temp_5 = []
        temp_4 = []
        temp_3 = []

        for w in vote.keys():
            if vote[w] == 5 and flag_5:
                temp_5.append(w)
            elif vote[w] >= 4 and flag_4:
                temp_4.append(w)
                flag_5 = False
            elif vote[w] >= 3:
                flag_5 = False
                flag_4 = False
                temp_3.append(w)

        if len(temp_5) != 0:
            # print("Five")
            temp = temp_5
        elif len(temp_4) != 0:
            # print("four")
            temp = temp_4
        elif len(temp_3) != 0:
            # print("Three")
            temp = temp_3
        else:
            break
        noise_u = np.zeros((0, n))
        flip_u = np.zeros((0, n))
        for u in temp:
            noise_u = np.r_[noise_u, train[u, :].reshape((1, n))]
            suspected_noise_dic[u] += 1
            train_y[u] = predict_res[u]

            # flip_u = np.r_[flip_u, train[u, :].reshape((1, n))]
            # if train[u, 0] == 1:
            #     train[u, 0] = 0  # todo  画图改成了0
            # else:
            #     train[u, 0] = 1
        print("T:", T, "length", len(temp))

        if len(temp) == len(last):  # 比较两次是否一样的 一样的就停止下来
            sorted(temp)
            sorted(last)
            for s in range(len(last)):
                if temp[s] != last[s]:
                    break
                if s == len(last) - 1 and temp[s] == last[s]:
                    break_flag = True
        if break_flag or len(temp) == 0:
            break

        last = copy.deepcopy(temp)

    for q in vote.keys():
        if suspected_noise_dic[q] % 2 == 1:
            suspected_noise.append(q)

    return suspected_noise,predict_res



# todo 这里添加五折交叉验证
def cross_all_five_return_index(traindata, splits=5):
    # np.random.shuffle(traindata)
    m,n=traindata.shape

    dic = {}
    #将predict_res组合起来，找到标签翻转的目标
    pred_res_cross=np.ones((m,splits))*(-999)
    for i in range(m):
        dic[i] = 0
    # todo 5折
    folds = StratifiedKFold(n_splits=splits, shuffle=True)
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(traindata[:,1:],traindata[:,0])):
        tr = traindata[trn_idx]
        trn = pd.DataFrame(tr)
        val = traindata[val_idx]
        copy_train = copy.deepcopy(trn)
        noise,predict_res=GNN_1(copy_train)
        # noise,predict_res = RD_fuction_MNN_7(copy_train)  # 传出噪声点
        for ti in range(len(trn_idx)):
            pred_res_cross[trn_idx[ti],n_fold] = predict_res[ti]

        for i in noise:
            dic[trn_idx[i]] += 1

    flip_label_target=mode(pred_res_cross,axis=1)[0]

    #过滤+翻转
    '''
    edited:过滤后的样本
    flip: 标签需要翻转的样本
    noise:噪声样本，包括翻转和过滤
    '''
    edited = []
    flip = []
    noise = []
    noise_index = []
    for i in dic.keys():
        if dic[i]<=1:
            edited.append(traindata[i])
        elif dic[i] >=2:
            noise_index.append(i)
            noise.append(traindata[i])
            if dic[i] >=4:
                traindata[i, 0] = flip_label_target[i]
                flip.append(traindata[i])
                edited.append(traindata[i])

    edited_train = array(edited)
    flip = array(flip)
    noise = array(noise)
    #traindata=edit+noise-flip
    return edited_train,flip,noise,noise_index

def main_train_return_index(data):
    # train_add_noise = load_data(data, noise)
    # train = copy.deepcopy(train_add_noise)
    delete_noise_train, flip, noise, kmnn_noise_index = cross_all_five_return_index(data)  # todo   crfnfl_all_ten

    return delete_noise_train, flip, noise, kmnn_noise_index

def test_eirs():
    # data_file_path = r'D:\09 Exp\kNN_label_noise\kNN_label_noise\dataset\datasplit_csv\multiclass\\'
    # data_train=pd.read_csv(data_file_path+'noisy_train_iris_0.3.csv',header=None).values
    data_file_path = r'D:\09 Exp\kNN_label_noise\kNN_label_noise\dataset\datasplit_csv\\'
    data_train=pd.read_csv(data_file_path+'noisy_train_fourclass_0.3.csv',header=None).values
    delete_noise_train, flip, noise, kmnn_noise_index=main_train_return_index(data_train)
    print(flip)

if __name__ == '__main__':
    test_eirs()
