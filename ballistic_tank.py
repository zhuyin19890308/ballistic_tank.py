#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Zhu Yin
@license: (C) Copyright 2018, USTC IMR.
@contact: zzyIzzf@mail.ustc.edu.cn
@software: Pycharm
@file: ballistic.py
@time: 2021/4/17 15:35
@desc: 坦克装甲武器的弹道解算
"""
import random
import time
import numpy as np
import commonPara
import pandas as pd
from decimal import Decimal


class ballistic():
    def __init__(self, bulletName, theta0, Cb, **kwargs):
        """
        类初始化函数
        :param bulletName: 炮弹名称
        :param theta0: 炮弹初始射角
        :param T: 当地温度
        :param Wz: 横风
        :param Wy: 竖风
        :param Wx: 纵风
        :param i: 弹形系数
        :param m: 弹质量
        :param d: 弹直径
        :param Cb: 弹道系数
        :param P0: 地面气压值
        :param shootAxis: 炮目坐标方位角
        :param xw: 世界坐标系中车体坐标 x
        :param yw: 世界坐标系中车体坐标 y
        :param zw: 世界坐标系中车体坐标 z
        :param isSeparate: 是否加入散布
        :param E_theta_x: 射角误差
        :param E_theta_z: 方位向误差
        :param E_v0: 初速误差
        :param E_cb: 弹道系数误差
        """
        global currentKwargs
        # 判断参数字典是否合法
        legalParaDic = {"T": 0, "Wz": 0, "Wy": 0, "Wx": 0, "i": 0, "m": 0, "d": 0, "P0": 0,"shootAxis": 0,
                        "xw": 0, "yw": 0, "zw": 0, "isSeparate": False, 
                        "E_theta_x": 0, "E_theta_z": 0, "E_v0": 0, "E_cb": 0}
        for tmp in enumerate(kwargs):
            currentKwargs = tmp[1]
            try:
                legalParaDic[currentKwargs]
            except:
                print("Unkwon arg:", currentKwargs)
            else:
                legalParaDic[currentKwargs] = kwargs[tmp[1]]

        self.isSeparate = np.array(legalParaDic["isSeparate"]).astype(np.int)
        # 误差
        self.E_theta_x = legalParaDic["E_theta_x"]
        self.E_theta_z = legalParaDic["E_theta_z"]
        self.E_v0 = legalParaDic["E_v0"]
        self.E_cb = legalParaDic["E_cb"]
        # 此处加上的初速度的误差 # 这是初始值
        self.v0 = commonPara.bulletTypicalSpeedDic[bulletName] + self.isSeparate * self.genRandomFeed(0, self.E_v0)
        self.bulletName = bulletName
        # 此处加上了误差
        self.theta0 = theta0 + self.isSeparate * self.genRandomFeed(0, self.E_theta_x)
        # 此处加上了误差
        self.Cb = Cb + self.isSeparate * self.genRandomFeed(0, self.E_cb)
        self.T = legalParaDic["T"]
        self.Wz = legalParaDic["Wz"]
        self.Wy = legalParaDic["Wy"]
        self.Wx = legalParaDic["Wx"]
        self.i = legalParaDic["i"]
        self.m = legalParaDic["m"]
        self.d = legalParaDic["d"]
        self.P0 = legalParaDic["P0"]
        self.shootAxis = legalParaDic["shootAxis"]
        # 速度初始化，此处乘上了方位向的误差
        self.vx = self.v0 * np.cos(self.isSeparate * self.genRandomFeed(0, self.E_theta_z)) * np.cos(
            commonPara.angle2rad(theta0))
        self.vy = self.v0 * np.cos(self.isSeparate * self.genRandomFeed(0, self.E_theta_z)) * np.sin(
            commonPara.angle2rad(theta0))
        self.vz = self.v0 * np.sin(self.isSeparate * self.genRandomFeed(0, self.E_theta_z))
        self.xw = legalParaDic["xw"]
        self.yw = legalParaDic["yw"]
        self.zw = legalParaDic["zw"]

        self.fun_value = [[], [], []]  # 微分方程的值
        #
        self.speed = [[], [], [], [], [], [], [], [], [], []]
        # 默认速度设定
        self.initRetValues()

    def genRandomFeed(self, mu, sigma):
        """
        产生随机数，高斯分布
        :param mu: 期望
        :param sigma: 方差
        :return:
        """
        return random.gauss(mu, sigma)

    def initRetValues(self):
        # 最一开始的情况下，默认发射平面就是XOY面，即横向速度为0
        self.speed[0].append(self.vx)  # 初始水平速度
        self.speed[1].append(self.vy)  # 初始垂直速度
        self.speed[2].append(self.vz)  # 初始横向速度，该速度受横风影响
        self.speed[3].append(self.Wx)  # 初始风速 wx
        self.speed[4].append(self.Wy)  # 初始风速 wy
        self.speed[5].append(self.Wz)  # 初始风速 wz
        self.speed[6].append(self.v0)  # 初始速度 vr
        self.speed[7].append(0)  # 弹道高度 x
        self.speed[8].append(0)  # 弹道高度 y
        self.speed[9].append(0)  # 弹道高度 z

        if self.v0 == 0:
            print("invalid initial speed v0!")
            self.v0 = input("v0=")

    # 以下定义微分方程组中用到的函数
    def fun_II(self, y, isStandardP=True):
        """
        气压函数
        :param y: 高度
        :param isStandardP: 如果是False，则表示返回的气压函数需要换算，否则按照标准气压条件返回值
        :return:
        """
        ret = 0
        if y <= 9300:
            ret = pow(1 - 2.1905 * pow(10, -5) * y, 5.4)
            # print("ret", ret)
        elif y > 9300 and y <= 12000:
            ret = 0.29228 * np.exp(-2.12064 * (np.arctan((2.344 * (y - 9300) - 6328) / 32221) + 0.193925))
        elif y > 12000:
            ret = 0.19372 * np.exp(-1 * (y - 12000) / 6483.3)

        # 判断当前的气压条件，给定返回值
        if isStandardP:
            return ret
        else:
            if self.P0 == 0:
                print("Check P0!")
                return False
            return ret * self.P0 / commonPara.commonParaDic["Pon"]

    def fun_Cb(self, isBlind=False):
        """
        求弹道系数
        :param isBlind: 如果是False则利用求解的方法，如果是True则是取值的方法
        :return:弹道系数
        """
        if isBlind:
            return self.Cb
        else:
            if self.i == 0 or self.m == 0 or self.d == 0:
                print("Check ammo property!")
                return False
            Cb = self.i * pow(self.d, 2) * pow(10, 3) / self.m
            return Cb

    def soundSpeed(self, y):
        """
        计算声速
        :param y: 高度
        :return: 返回声速
        """
        Tvon = commonPara.commonParaDic["virtualTemp_ground"]
        Tv1 = Tvon - commonPara.commonParaDic["G1"] * y
        return np.sqrt(commonPara.commonParaDic["k"] * commonPara.commonParaDic["R"] * Tv1)

    def fun_Fv(self, vr, C1):
        # Cson = 341.1
        # vt2 = (vr / C1) * Cson
        vt = vr * np.sqrt(288.9 / (288.9 - commonPara.commonParaDic["G1"] * self.speed[8][-1]))
        ret = 0
        if vt < 250:
            ret = 0.00007454 * pow(vt, 2)
        elif vt >= 250 and vt < 400:
            ret = 629.61 - 6.0255 * vt + 1.8756 * pow(10, -2) * pow(vt, 2) - 1.8613 * pow(10,
                                                                                          -5) * pow(vt, 3)
        elif vt >= 400 and vt <= 1400:
            ret = 6.394 * pow(10, -8) * pow(vt, 3) - 6.325 * pow(10, -5) * pow(vt,
                                                                               2) + 0.1548 * vt - 26.63
        elif vt > 1400:
            ret = 0.00012315 * pow(vt, 2)
        return ret

    def fun_Gv(self, vr, C1):
        # Cson = 341.1
        # vt = (vr / C1) * Cson

        vt = vr * np.sqrt(288.9 / (288.9 - commonPara.commonParaDic["G1"] * self.speed[8][-1]))
        Gv = self.fun_Fv(vr, C1) / vt
        return Gv

    def fun_cal_vr(self, vx, vy, vz):
        vr = np.sqrt(pow(vx - self.Wx, 2) + pow(vy - self.Wy, 2) + pow(vz - self.Wz, 2))
        return vr

        # 以下定义微分方程组

    def fun1(self, y, vr):
        # Cb
        cb = self.fun_Cb(True)
        # H(y)
        II = self.fun_II(y, True)
        # print("II", II)
        Tvon = commonPara.commonParaDic["virtualTemp_ground"]
        Tv1 = Tvon - commonPara.commonParaDic["G1"] * y
        hy = II * (Tvon / Tv1)

        # G(V,C1)
        C1 = self.soundSpeed(y)
        gv = self.fun_Gv(vr, C1)

        # -1 * self.fun_Cb(True) * self.fun_II(y, True) * (commonPara.commonParaDic["virtualTemp_ground"] / (
        #             commonPara.commonParaDic["virtualTemp_ground"] - 0.006328 * y)) * self.fun_Gv(vr, self.soundSpeed(y)) * (self.speed[0][-1] - self.Wx)
        # 存入到方程的值的函数中去
        # print("cb", cb)
        # print("hy", hy)
        # print("gv", gv)
        # print("deltaV", self.speed[0][-1] - self.Wx)
        self.fun_value[0].append(-1 * cb * hy * gv * (self.speed[0][-1] - self.Wx))
        return self.fun_value[0][-1]

    def fun2(self, y, vr):
        # Cb
        cb = self.fun_Cb(True)
        # H(y)
        II = self.fun_II(y, True)
        Tvon = commonPara.commonParaDic["virtualTemp_ground"]
        Tv1 = Tvon - commonPara.commonParaDic["G1"] * y
        hy = II * (Tvon / Tv1)

        # G(V,C1)
        C1 = self.soundSpeed(y)
        gv = self.fun_Gv(vr, C1)

        # -1 * self.fun_Cb(True) * self.fun_II(y, True) * (commonPara.commonParaDic["virtualTemp_ground"] / (
        #             commonPara.commonParaDic["virtualTemp_ground"] - 0.006328 * y)) * self.fun_Gv(vr, self.soundSpeed(y)) * (self.speed[0][-1] - self.Wx)
        # 存入到方程的值的函数中去
        self.fun_value[1].append(-1 * cb * hy * gv * (self.speed[1][-1] - self.Wy) - commonPara.commonParaDic["g"])
        return self.fun_value[1][-1]

    def fun3(self, y, vr):
        # Cb
        cb = self.fun_Cb(True)
        # H(y)
        II = self.fun_II(y, True)
        Tvon = commonPara.commonParaDic["virtualTemp_ground"]
        Tv1 = Tvon - commonPara.commonParaDic["G1"] * y
        hy = II * (Tvon / Tv1)

        # G(V,C1)
        C1 = self.soundSpeed(y)
        gv = self.fun_Gv(vr, C1)

        # -1 * self.fun_Cb(True) * self.fun_II(y, True) * (commonPara.commonParaDic["virtualTemp_ground"] / (
        #             commonPara.commonParaDic["virtualTemp_ground"] - 0.006328 * y)) * self.fun_Gv(vr, self.soundSpeed(y)) * (self.speed[0][-1] - self.Wx)
        # 存入到方程的值的函数中去
        self.fun_value[2].append(-1 * cb * hy * gv * (self.speed[2][-1] - self.Wz))
        return self.fun_value[2][-1]

    def fun_dxdt(self, vx):
        # return self.speed[0][-1]
        return vx

    def fun_dydt(self, vy):
        # return self.speed[1][-1]
        return vy

    def fun_dzdt(self, vz):
        # return self.speed[2][-1]
        return vz

    def fun_RK4(self):
        time = 0
        time_interval = 0.001
        isTouchGround = False  # 是否落地
        while not isTouchGround:
            K1 = time_interval * self.fun1(self.speed[8][-1], self.speed[6][-1])
            G1 = time_interval * self.fun2(self.speed[8][-1], self.speed[6][-1])
            H1 = time_interval * self.fun3(self.speed[8][-1], self.speed[6][-1])
            X1 = time_interval * self.speed[0][-1]
            Y1 = time_interval * self.speed[1][-1]
            Z1 = time_interval * self.speed[2][-1]

            K2 = time_interval * (0.5 * K1 + self.fun1(self.speed[8][-1], self.speed[6][-1]))
            G2 = time_interval * (0.5 * G1 + self.fun2(self.speed[8][-1], self.speed[6][-1]))
            H2 = time_interval * (0.5 * H1 + self.fun3(self.speed[8][-1], self.speed[6][-1]))
            X2 = time_interval * (self.speed[0][-1] + 0.5 * X1)
            Y2 = time_interval * (self.speed[1][-1] + 0.5 * Y1)
            Z2 = time_interval * (self.speed[2][-1] + 0.5 * Z1)

            K3 = time_interval * (0.5 * K2 + self.fun1(self.speed[8][-1], self.speed[6][-1]))
            G3 = time_interval * (0.5 * G2 + self.fun2(self.speed[8][-1], self.speed[6][-1]))
            H3 = time_interval * (0.5 * H2 + self.fun3(self.speed[8][-1], self.speed[6][-1]))
            X3 = time_interval * (self.speed[0][-1] + 0.5 * X2)
            Y3 = time_interval * (self.speed[1][-1] + 0.5 * Y2)
            Z3 = time_interval * (self.speed[2][-1] + 0.5 * Z2)

            K4 = time_interval * (self.fun1(self.speed[8][-1], self.speed[6][-1]) + K3)
            G4 = time_interval * (self.fun2(self.speed[8][-1], self.speed[6][-1]) + G3)
            H4 = time_interval * (self.fun3(self.speed[8][-1], self.speed[6][-1]) + H3)
            X4 = time_interval * (self.speed[0][-1] + X3)
            Y4 = time_interval * (self.speed[1][-1] + Y3)
            Z4 = time_interval * (self.speed[2][-1] + Z3)

            value1 = self.speed[0][-1] + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            value2 = self.speed[1][-1] + (G1 + 2 * G2 + 2 * G3 + G4) / 6
            value3 = self.speed[2][-1] + (H1 + 2 * H2 + 2 * H3 + H4) / 6

            self.speed[0].append(value1)
            self.speed[1].append(value2)
            self.speed[2].append(value3)
            self.speed[6].append(self.fun_cal_vr(self.speed[0][-1], self.speed[1][-1], self.speed[2][-1]))

            # self.speed[7].append(self.speed[7][-1] + self.speed[0][-1] * time_interval)
            # self.speed[8].append(self.speed[8][-1] + self.speed[1][-1] * time_interval)
            # self.speed[9].append(self.speed[9][-1] + self.speed[2][-1] * time_interval)
            self.speed[7].append(self.speed[7][-1] + (X1 + 2 * X2 + 2 * X3 + X4) / 6)
            self.speed[8].append(self.speed[8][-1] + (Y1 + 2 * Y2 + 2 * Y3 + Y4) / 6)
            self.speed[9].append(self.speed[9][-1] + (Z1 + 2 * Z2 + 2 * Z3 + Z4) / 6)

            time = time + time_interval
            if self.speed[8][-1] <= 0 or self.speed[0][-1] <= 0:
                isTouchGround = True

    # def test_RK4(self):

    # 取射表值:x
    def get_X_distance(self):
        x_distance_list = []
        for singleList in commonPara.bulletFireTableDic[self.bulletName]:
            x_distance_list.append(singleList[2])
        return x_distance_list

    # 取射表值:y
    def get_Y_distance(self):
        y_distance_list = []
        for singleList in commonPara.bulletFireTableDic[self.bulletName]:
            y_distance_list.append(singleList[3])
        return y_distance_list

    # 计算误差
    # x方向弹道高，射表与计算值误差
    def cal_X_postion_error(self, X_list, X_list_shebiao, var):
        # 已有真值长度，计算长度可能比较这个多
        shebiao_Len = len(X_list_shebiao)
        if len(X_list) < shebiao_Len:
            print("Calculated length of value must be longer than fire table")
            return False

    # y方向弹道高，射表与计算值误差
    def cal_Y_postion_error(self, ):
        pass

    def angle2miwei(self, angle):
        """
        角度转密位
        :param angle: 角度
        :return:
        """
        return angle * 6000 / 360

    def miwei2angle(self, miwei):
        """
        密位转角度
        :param miwei:
        :return:
        """
        return miwei * 360 / 6000

    def decodeWeatherMessage(self, message):
        """
        炮兵气象信息解码函数,解算出来的高度已经加上了高程信息
        :param message: 气象通报数据
        :return: 返回解码值字典  decodedMsgDic
        """
        decodedMsgDic = {}
        # 剔除非法信息
        if len(message) == 0:
            print('Invalid weather message!')
            return
        messageList = message.split("-")
        # 解码气象通报类型
        messageTypeDic = {"1111": "手工作业气象通报",
                          "2222": "防空兵气象通报",
                          "3333": "声测气象通报",
                          "4444": "防化气象通报",
                          "5555": "计算机气象通报"}
        messageType = messageTypeDic[messageList[0]]  # 气象类型
        messageTime = messageList[1][:2] + "时" + messageList[1][2::] + "分"  # 时间
        messageHeight = self.minusDetect(messageList[2])  # 气象站高程
        messagePressureOffset = self.minusDetect(messageList[3][:3])  # 气象站地面气压偏差
        messageTemperatureOffset = self.minusDetect(messageList[3][3::])  # 气象站地面气温偏差

        messageCodeDic = {}  # 保存气象数据
        heightInfo = 0
        temperatureOffset = 0
        windAxis = 0
        windSpeed = 0

        for i in messageList[4:]:  # 遍历索引
            if len(i) == 2:
                heightLabel = int(i)
                if heightLabel < 10:
                    heightInfo = heightLabel * 100  # 小于10时，以百为单位。例如02表示200米
                elif heightLabel >= 10:
                    heightInfo = heightLabel * 1000  # 大于等于10时，以千为单位。例如12表示12000米
                heightInfo = heightInfo + messageHeight  # 加上气象站高程信息
            else:
                if i[:2] == "99":
                    temperatureOffset = "unknow"  # 弹道气温偏差
                else:
                    temperatureOffset = i[:2]  # 弹道气温偏差
                windAxis = int(i[2:4]) * 100  # 风向角
                # windOriation = abs(self.shootAxis - windAxis)  # 风角=炮目坐标方位角-风向坐标方位角
                windSpeed = i[4:6]
            messageCodeDic[str(heightInfo)] = [temperatureOffset, windAxis, windSpeed]
        decodedMsgDic["messageType"] = messageType
        decodedMsgDic["messageTime"] = messageTime
        decodedMsgDic["messageHeight"] = messageHeight
        decodedMsgDic["messagePressureOffset"] = messagePressureOffset
        decodedMsgDic["messageTemperatureOffset"] = messageTemperatureOffset
        decodedMsgDicRet = decodedMsgDic.copy()
        decodedMsgDicRet.update(messageCodeDic)
        # print(decodedMsgDicRet)
        return decodedMsgDicRet

    def minusDetect(self, num):
        """
        只能接受数值，不接受字符串
        规定以5开头为负数，该函数用于检测是否是负数
        :param num:
        :return:
        """
        num = num[::-1]
        if num[-1] == '5':
            num = num[:-1][::-1]
            return -1 * int(num)
        else:
            return int(num[::-1])

    def writeData(self, path="./"):
        # 先将数据转置
        new_matrix = []
        print(len(self.speed))
        for i in range(len(self.speed[7])):
            listTmp = []
            for speedLen in range(len(self.speed)):
                if speedLen == 3 or speedLen == 4 or speedLen == 5:
                    continue
                listTmp.append(self.speed[speedLen][i])  # 将speed二维列表中对应的数据整到一个列表中
            new_matrix.append(listTmp)
        # print(new_matrix)
        fileTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        filePath = path + fileTime + ".txt"
        output = open(filePath, 'w', encoding='gbk')
        output.write("弹种："+str(self.bulletName)+"\t"
                     +"弹道系数："+str(self.Cb)+"\t"
                     +"炮弹初始射角"+str(self.theta0)+"\n")
        output.write('初始水平速度 \t 初始垂直速度 \t 初始横向速度 \t 初始速度 \t 弹道高度x \t 弹道高度y \t 弹道高度z \n')
        for i in range(len(new_matrix)):
            # output.writelines(str(new_matrix[i]).replace("["," ").replace("]"," ").replace(",","        "))
            for j in range(len(new_matrix[i])):
                output.write(str(round(new_matrix[i][j], 10)))  # write函数不能写int类型的参数，所以使用str()转化
                output.write('\t')  # 相当于Tab一下，换一个单元格
            output.write('\n')  # 写完一行立马换行
        print("保存文件成功")
        output.close()
