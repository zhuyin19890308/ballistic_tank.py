#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Zhu Yin
@license: (C) Copyright 2021
@contact: zzyIzzf@mail.ustc.edu.cn
@software: Pycharm
@file: testBallistic.py
@time: 2021/4/26 20:58
@desc:
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

# bullet = ballistic("榴", 1, 15, Cb=0.0291)  # 0.0291  0.302 0.232 0.295
bullet = ballistic("105穿", commonPara.miwei2angle(8.29), 0.295, T=288.9, Wz=10)  # 0.0291  0.302
bullet.fun_RK4()
#
plt.figure(1, figsize=(8, 2))
plt.style.use("seaborn")
plt.plot(bullet.speed[7], bullet.speed[9], label="y_postion", linestyle="-")

print("最大弹道高：", max(bullet.speed[8]))
print("射距：", max(bullet.speed[7]))
print("最大弹道高存速：", bullet.speed[6][bullet.speed[8].index(max(bullet.speed[8]))])
print("最大射距存速：", bullet.speed[6][-1])
print("横风偏移量", bullet.speed[9][-1])

print(len(bullet.speed[7]))
plt.legend()
plt.savefig("./fig.png", dpi=300)
plt.show()

X = bullet.speed[7]
Y = bullet.speed[8]
Z = bullet.speed[9]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Z, X, Y, label="ballistic")

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.001)
