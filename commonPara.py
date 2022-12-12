#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Zhu Yin
@license: (C) Copyright 2018, USTC IMR.
@contact: zzyIzzf@mail.ustc.edu.cn
@software: Pycharm
@file: commonPara.py
@time: 2021/4/17 15:40
@desc:本脚本用于定义通用参数
"""

import numpy as np
import pyqt5_tools as qt
import matplotlib.pyplot as plt
import fireTable

"""
PI:圆周率
g:重力加速度 m/s²
virtualTemp_ground:底面虚温K
Pon:底面标准气压值kPa
k:空气绝热指数
R:空气气体常数 J/(kg*K)
Omiga: rad/s 地球自转角速度，这个角速度在转换后只需要用其标量，矢量方向去除信息
R_earth:地球半径  米  平均半径
"""

commonParaDic = {"PI": np.pi, "g": 9.80665, "virtualTemp_ground": 288.900000,
                 "Pon": 100.000000, "k": 1.400000, "R": 289.000000, "G1": -0.006328,
                 "Omiga":7.292115*pow(10,-5), "R_earth":6371393}

# 制式弹的；类型
bulletTypeDic = {"105穿": 1, "榴": 2, "破": 3}
# 制式弹的参数 列表内参数顺序为hr d E
bulletTypeShapeDic = {1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]}
# 制式弹的初速
bulletTypicalSpeedDic = {"105穿": 1580, "榴": 900, "破": 1200,"120杀爆榴弹(全号)":425,
                            "120杀爆榴弹(6号)":349,"120杀爆榴弹(5号)":314,"120杀爆榴弹(4号)":276,
                            "120杀爆榴弹(3号)":230,"120杀爆榴弹(2号)":185,}
# 制式弹与射表的对应
bulletFireTableDic = {"105穿": 0, "榴": fireTable.liu100, "破": 0}
# 龙格库塔方法微分步长计算方法
dt = 0.001  # 微分步长
flyTime = []  # 炮弹总飞行时间


def setFlyTime(time):
    timeRange = np.arange(0, time + dt, dt)
    flyTime = timeRange
    print("飞行时间为：", flyTime)


def calBulletShapePara(bulletType=1):
    """
    按照经验公式计算弹形系数
    :param bulletType: 已有制式弹的外型
    :return: 返回弹形系数
    """
    H = 0
    i = 0  # 弹形系数
    for type in enumerate(bulletTypeDic):
        # print(type[1])
        # print(bulletTypeDic[type[1]])
        if bulletType is bulletTypeDic[type[1]]:
            print("当前的基础制式弹形为：", type[1])
            hr = bulletTypeShapeDic[bulletType][0]
            d = bulletTypeShapeDic[bulletType][1]
            E = bulletTypeShapeDic[bulletType][2]
            H = (hr + d) / E - 0.30
            i = 2.90 - 1.373 * H - 0.32 * H * H - 0.0267 * H * H * H
            print("弹形系数为：", i)
        else:
            pass
    return i


def f(y):
    """
    定义导数方程
    :param y: 变量
    :return:
    """
    # 待定义
    return 0


def ode_rk(t, y0):
    """
    四阶龙格库塔法
    :param t:
    :param y0:
    :return:
    """
    # 定义初始变量
    N = len(t)
    y = np.zeros(N, 1)
    y[0] = y0

    if len(flyTime) == 0:
        return
    else:
        for n in range(0, N - 1):
            h = flyTime[n + 1] - flyTime[n]
            k1 = f(flyTime[n])
            k2 = f(flyTime[n] + h / 2 * k1)
            k3 = f(flyTime[n] + h / 2 * k2)
            k4 = f(flyTime[n] + h * k3)
            y[n + 1] = y[n] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return


def angle2miwei(angle):
    """
    角度转密位
    :param angle: 角度
    :return:
    """
    return angle * 6000 / 360


def miwei2angle(miwei):
    """
    密位转角度
    :param miwei:
    :return:
    """
    return miwei * 360 / 6000


def angle2rad(angle):
    """
    角度转弧度
    :param angle:
    :return:
    """
    return angle * np.pi / 180

def rad2angle(rad):
    """
    弧度转角度
    :param rad: 弧度值
    :return: 角度
    """
    return 180*rad/np.pi