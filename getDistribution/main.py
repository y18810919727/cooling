# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:16:29 2021

@author: qhr
"""
import math
import csv
import os

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xlsxwriter as xw
# from util import SimpleLogger, show_data, init_weights, array_operate_with_nan, get_Dataset, visualize_prediction, t2np, \
#     draw_table, draw_table_all, calculation_ms
from pandas import Series

from util import get_Dataset


def xw_toExcel(data, fileName):  # xlsxwriter库储存数据到excel
    workbook = xw.Workbook(fileName)  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    worksheet1.write_row('A1', data[1])  # 从A1单元格开始写入表头
    worksheet1.write_row('A2', data[2])  
    worksheet1.write_row('A3', data[3])  
    worksheet1.write_row('A4', data[4])  
    worksheet1.write_row('A5', data[5])  
    workbook.close()  # 关闭表


def box_img(states,no,dataset):
    figure, axes = plt.subplots()
    axes.set_xticklabels(dataset)
    axes.boxplot(states[no], patch_artist=False)  # 描点上色
    plt.title('state%d'%(no))
    plt.xlabel('Pserver(w)')
    plt.ylabel('time(s)')
    axes.yaxis.grid(True)
    plt.savefig('./results/state%d.png' % no)
    #plt.show()


# 按照固定区间长度绘制频率分布直方图
# bins_interval 区间的长度
# margin        设定的左边和右边空留的大小
def probability_distribution(data, bins_interval=1, margin=1,no=0):
    plt.rcParams['font.sans-serif'] = ['SimHei']				# 解决中文无法显示的问题
    bins = range(min(data), max(data) + bins_interval - 1, bins_interval)
    xmin=min(data)-min(data)%bins_interval
    xmax=max(data)+bins_interval-max(data)%bins_interval

    plt.figure(figsize=(7,6))
    plt.xlim(xmin , xmax)
    plt.title("state "+str(no)+" frep")
    plt.title("Frequency of duration of state %s" % str(no))
    plt.xlabel('Time')
    plt.ylabel('frequency')
    # 频率分布normed=True，频次分布normed=False
    prob,left,rectangle = plt.hist(x=data, bins=bins, density=True, histtype='bar', color=['r'])
    for x, y in zip(left, prob):
        # 字体上边文字
        # 频率分布数据 normed=True
        if y>0:
            plt.text(x + bins_interval / 2, y + 0.0015*5/bins_interval, '%.2f' % (y*bins_interval), ha='center', va='top')
        # 频次分布数据 normed=False
        # plt.text(x + bins_interval / 2, y + 0.25, '%.2f' % y, ha='center', va='top')
    plt.yticks([])
    plt.savefig('./results/state%d.png' % no)
    plt.show()

#读取csv文件
if __name__ == '__main__':
    #file = 'lable_all.csv'
    dataset=['P-1.7k','P-3.8k','P-4.2k','P-6.3k']
    all_states = {}
    for everdata in dataset:
        # Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in get_Dataset('../middata/'+everdata+'.csv')]
        # data = pd.read_csv(r'../middata/'+everdata+'.csv')
        # ser = Series(strain.squeeze(-1).astype(int))
        # data['states'] = ser
        # data.to_csv(r'lable_%s.csv'%(everdata), mode='a', index=False)

        states=[ [], [], [], [], [], [] ]
        average=[]
        StandardDeviation=[]
        fourdivision=[]
        n=0

        nowState="0"
        with open('lable_%s.csv'%(everdata), "r") as f:
            reader = csv.reader(f)
            for i,row in enumerate(reader):
                if i>=1:
                    if row[10]!=nowState:
                        states[int(nowState)].append(n)
                        n = 0
                        nowState = str((int(nowState)+1)%6) #row[10]
                        if row[10]==nowState:
                            n = n + 1
                    else:
                        n = n + 1
        states[int(nowState)].append(n)

        for i in range(4):
            #把第一轮写入的数据删掉
            states[i].pop(0)
        all_states[everdata]=states

    # for i in states:
    #     #计算均值
    #     count = 0
    #     fc = 0
    #     for j in i:
    #         count = count + j
    #     average.append(count/ (len(i)))
    #     #计算标准差
    #     for k in i:
    #         fc = fc + (k-average[-1])**2
    #     StandardDeviation.append(math.sqrt(fc/(len(i))))
    #     #求四分位点
    #     fourdivision.append([sorted(i)[(len(i)-1)//4],sorted(i)[(len(i)-1)*3//4]])
    #
    # print("Mean:", average)
    # print("std", StandardDeviation)
    # print("quartiles", fourdivision)

    #画箱型图

    states = {}
    for i in range(1, 5):
        states[i] = [];
    # 得到画板、轴
    for key, value in all_states.items():
        for i in range(1, 5):
            states[i].append(value[i])



    #画前3个频率直方图
    # probability_distribution(data=states[1], bins_interval=5,margin=0,no=1)
    # probability_distribution(data=states[2], bins_interval=1,margin=0,no=2)
    # probability_distribution(data=states[3], bins_interval=1,margin=0,no=3)
    # probability_distribution(data=states[4], bins_interval=1,margin=0,no=4)
    # #probability_distribution(data=states[5], bins_interval=1,margin=0,no=5)
    for i in range(1, 5):
        box_img(states,i,dataset)




    # fileName = 'state.xlsx'
    # xw_toExcel(states, fileName)