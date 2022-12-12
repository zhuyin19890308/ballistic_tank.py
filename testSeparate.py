#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Zhu Yin
@license: (C) Copyright 2021, 207研究所.
@contact: zzyIzzf@mail.ustc.edu.cn
@software: Pycharm
@file: testSeparate.py
@time: 2021/5/7 20:42
@desc: 本脚本用于检验散布模型
"""
from mpl_toolkits.mplot3d import axes3d
from ballistic_tank import *
import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLineEdit
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator
import sys
from mayavi import mlab

result_x = []
result_y = []
result_z = []
totalStep = 101
stanrdDistance_bullet = ballistic("105穿", commonPara.miwei2angle(3.94), 0.301)
stanrdDistance_bullet.fun_RK4()
stanrdDistance = stanrdDistance_bullet.speed[7][-1]
for i in range(totalStep):
    bullet = ballistic("105穿", commonPara.miwei2angle(3.94), 0.301,
                       isSeparate=True,
                       E_theta_x=2,
                       E_theta_z=0.0001,
                       E_v0=3,
                       E_cb=0.001)
    bullet.fun_RK4()
    result_x.append(bullet.speed[7][-1] - stanrdDistance)
    result_y.append(bullet.speed[8][list(abs(bullet.speed[7]-stanrdDistance)).index(min(list(abs(bullet.speed[7]-stanrdDistance))))])
    result_z.append(bullet.speed[9][-1])
    # print("射距：", max(bullet.speed[7]))
    print("进度", i*100/(totalStep-1),"%")
print(result_x)
print(result_z)
plt.figure()
plt.scatter(result_x, result_z)
plt.show()
