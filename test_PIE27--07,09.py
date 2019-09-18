# -*- encoding:utf-8 -*-

'''
一个源域：
PIE27：类别分别为1-20
两个目标域（公共类4个）
PIE07：类别分别为1-9
PIE09：类别分别为6-14
值得注意的是：目标域总类是源域的子类
'''

import numpy as np
import scipy.io
import source_domain,target_domain,RandIndex,source,target
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn import preprocessing

classnum_PIE27=20; classnum_PIE07=9; classnum_PIE09=9
M=2   #目标域个数

PIE27=scipy.io.loadmat('PIE/27/PIE27.mat')
PIE07=scipy.io.loadmat('PIE/27/07,09/PIE07.mat')      #第一个目标域
PIE09=scipy.io.loadmat('PIE/27/07,09/PIE09(1).mat')      #第二个目标域

data_PIE27=PIE27['Xs']; data_PIE07=PIE07['Xt']; data_PIE09=PIE09['Xt']
label_27=PIE27['Ys']; label_07=PIE07['Yt']; label_09=PIE09['Yt']
U_PIE27=np.mat(PIE27['Us'])

'''降维'''
pca_27=KernelPCA(n_components=100,kernel='rbf')
pca_07=KernelPCA(n_components=100,kernel='rbf')
pca_09=KernelPCA(n_components=100,kernel='rbf')

data_PIE27=pca_27.fit_transform(data_PIE27)  #大小均为n*d'
data_PIE07=pca_07.fit_transform(data_PIE07)
data_PIE09=pca_09.fit_transform(data_PIE09)

'''预处理'''
data_PIE27=np.mat(preprocessing.scale(data_PIE27)).T   #大小均为d'*n
data_PIE07=np.mat(preprocessing.scale(data_PIE07)).T
data_PIE09=np.mat(preprocessing.scale(data_PIE09)).T

label_PIE27=[]; label_PIE07=[]; label_PIE09=[]
for i in range(len(label_27)):
    d=int(label_27[i])
    label_PIE27.append(d)
for i in range(len(label_07)):
    d=int(label_07[i])
    label_PIE07.append(d)
for i in range(len(label_09)):
    d=int(label_09[i])
    label_PIE09.append(d)

'''PIE27源域的WS'''
Lambda_PIE27=0.5
W_PIE27=source_domain.SLMC_W(data_PIE27,U_PIE27,Lambda_PIE27,classnum_PIE27)

Maxiter = 200; epsilon = 10**(-7); Lambda = 0.5
Y_PIE07 = np.mat(np.eye(classnum_PIE07, dtype=int))
Y_PIE09 = np.mat(np.eye(classnum_PIE09, dtype=int))
U_PIE07 = np.mat(source_domain.Normization(np.random.random((classnum_PIE07,data_PIE07.shape[1]))))
#np.savetxt('PIE/U_PIE(27)07.csv', U_PIE07, delimiter = ',')
U_PIE09=np.mat(source_domain.Normization(np.random.random((classnum_PIE09, data_PIE09.shape[1]))))
#np.savetxt('PIE/U_PIE(27)09.csv', U_PIE09, delimiter = ',')
U1 = U_PIE09; U2=U_PIE09

print("PIE07域")
J = J_new = float('inf')
for j in range(Maxiter):

    W1 = source.SLMC_W(data_PIE07, U1, Lambda, classnum_PIE07)
    U1 = source.SLMC_U(data_PIE07, classnum_PIE07, W1)
    J2 = 0
    J = J_new
    for k in range(classnum_PIE07):
        for i in range(data_PIE07.shape[1]):
            J2 = J2 + U1[k, i] ** 2 * np.linalg.norm((W1.T * data_PIE07[:, i] - Y_PIE07[:, k]), ord=2) ** 2
    J_new = 0.5 * np.linalg.norm(W1, ord=2) ** 2 + 0.5 * Lambda * J2
    # print(abs((J_new - J) / J_new))
    if (abs((J_new - J) / J_new) < epsilon):
        break
print(j)
Y1 = (W1.T * data_PIE07).T
raw_1, column_1 = Y1.shape
pre1 = []
for i in range(raw_1):
    _positon = np.argmax(Y1[i])
    m, n = divmod(_positon, column_1)
    pre1.append(n + 1)

NMI_1 = metrics.normalized_mutual_info_score(label_PIE07, pre1)
RI_1 = RandIndex.rand_index_score(label_PIE07, pre1)
print('NMI大小：',round(NMI_1,4))
print('RI：',round(RI_1,4))

print("PIE09域")
J = J_new = float('inf')
for j in range(Maxiter):

    W2 = source.SLMC_W(data_PIE09, U2, Lambda, classnum_PIE09)
    U2 = source.SLMC_U(data_PIE09, classnum_PIE09, W2)
    J2 = 0
    J = J_new
    for k in range(classnum_PIE09):
        for i in range(data_PIE09.shape[1]):
            J2 = J2 + U2[k, i] ** 2 * np.linalg.norm((W2.T * data_PIE09[:, i] - Y_PIE09[:, k]), ord=2) ** 2
    J_new = 0.5 * np.linalg.norm(W2, ord=2) ** 2 + 0.5 * Lambda * J2
    # print(abs((J_new - J) / J_new))
    if (abs((J_new - J) / J_new) < epsilon):
        break
print(j)
Y2 = (W2.T * data_PIE09).T
raw_2, column_2 = Y2.shape
pre2 = []
for i in range(raw_2):
    _positon = np.argmax(Y2[i])
    m, n = divmod(_positon, column_2)
    pre2.append(n + 1)
NMI_2 = metrics.normalized_mutual_info_score(label_PIE09, pre2)
RI_2 = RandIndex.rand_index_score(label_PIE09, pre2)
print('NMI大小：',round(NMI_2,4))
print('RI：',round(RI_2,4))

'''计算目标域的情况'''
print(' ')
print('目标域的情况')
Lambda_PIE07=1; Lambda_PIE09=1
beta = [10,50,100]
gama = [0.0001,0.001,0.01,0.1,1,10,50,100]
eta = 0.1
r = 15

D = np.mat(np.random.random((data_PIE07.shape[0], r)))  # 随机初始化公共字典
VT = [np.mat(np.random.random((classnum_PIE07, classnum_PIE27))),
      np.mat(np.random.random((classnum_PIE09, classnum_PIE27)))]
V = [np.mat(np.random.random((r, classnum_PIE07))),
     np.mat(np.random.random((r, classnum_PIE09)))]

NMI_t1_parameter = np.mat(np.zeros((8,8)))
NMI_t2_parameter = np.mat(np.zeros((8,8)))
RI_t1_parameter = np.mat(np.zeros((8,8)))
RI_t2_parameter = np.mat(np.zeros((8,8)))

for param_beta in range(len(beta)):
    for param_gama in range(len(gama)):
        gama_tar = [gama[param_gama], gama[param_gama]]
        print(' ')
        print('the value of parameter：', beta[param_beta], gama[param_gama])

        U_PIE07_tar = U_PIE07;
        U_PIE09_tar = U_PIE09;
        D_tar = D;
        V_tar = V;
        VT_tar = VT

        J_PIE07 = J_new_PIE07 = float("inf")
        J_PIE09 = J_new_PIE09 = float("inf")
        for m in range(Maxiter):
            W_PIE07 = target_domain.target_WTj(data_PIE07, U_PIE07_tar, W_PIE27, VT_tar[0], V_tar[0],
                                               D_tar, Lambda_PIE07, gama_tar[0], beta[param_beta])
            W_PIE09 = target_domain.target_WTj(data_PIE09, U_PIE09_tar, W_PIE27, VT_tar[1], V_tar[1],
                                               D_tar, Lambda_PIE09, gama_tar[1], beta[param_beta])
            U_PIE07_tar = target_domain.target_U(data_PIE07, classnum_PIE07, W_PIE07)
            U_PIE09_tar = target_domain.target_U(data_PIE09, classnum_PIE09, W_PIE09)
            VT_tar[0] = target_domain.Source_Target_VTj(W_PIE07, W_PIE27, beta[param_beta], eta)
            VT_tar[1] = target_domain.Source_Target_VTj(W_PIE09, W_PIE27, beta[param_beta], eta)
            WT = [W_PIE07, W_PIE09]
            try:
                D_tar = target_domain.Target_D(WT, V_tar, gama_tar, r)
            except Exception as e:
                print(e)
            try:
                D_tar = D_tar / (np.sum(D_tar, axis=0))
            except Exception as e:
                print(e)
            V_tar[0] = target_domain.Target_Vj(W_PIE07, D_tar, gama_tar[0], eta)
            V_tar[1] = target_domain.Target_Vj(W_PIE09, D_tar, gama_tar[1], eta)
            J_PIE07 = J_new_PIE07
            J_PIE09 = J_new_PIE09
            '''Dslr域的目标函数第二项的值'''
            J2_PIE07 = 0
            for k in range(classnum_PIE07):
                for i in range(data_PIE07.shape[1]):
                    J2_PIE07 = J2_PIE07 + U_PIE07_tar[k, i] ** 2 * np.linalg.norm(
                        (W_PIE07.T * data_PIE07[:, i] - Y_PIE07[:, k]), ord=2) ** 2

            '''Webcam域的目标函数第二项的值'''
            J2_PIE09 = 0
            for k in range(classnum_PIE09):
                for i in range(data_PIE09.shape[1]):
                    J2_PIE09 = J2_PIE09 + U_PIE09_tar[k, i] ** 2 * np.linalg.norm(
                        (W_PIE09.T * data_PIE09[:, i] - Y_PIE09[:, k]), ord=2) ** 2

            '''目标函数第三,四项的值'''
            J3 = 0
            for j in range(M):
                J3 = J3 + 0.5 * beta[param_beta] * np.linalg.norm(W_PIE27 - WT[j] * VT_tar[j], ord=2) ** 2
                + 0.5 * gama_tar[j] * np.linalg.norm(WT[j] - D_tar * V_tar[j], ord=2) ** 2
            classnum_Target = [classnum_PIE07, classnum_PIE09]
            '''目标函数第五，六项的值'''
            J5 = 0
            for j in range(M):
                V_temp = V_tar[j];
                VT_temp = VT_tar[j]
                for i in range(r):
                    J5 = J5 + np.linalg.norm(V_temp[i, :], ord=2)
                for i in range(classnum_Target[j]):
                    J5 = J5 + np.linalg.norm(VT_temp[i, :], ord=2)
                J5 = eta * J5

            J_new_PIE07 = 0.5 * (np.linalg.norm(W_PIE07, ord=2) ** 2 + Lambda_PIE07 * J2_PIE07) + J3 + J5
            J_new_PIE09 = 0.5 * (np.linalg.norm(W_PIE09, ord=2) ** 2 + Lambda_PIE09 * J2_PIE09) + J3 + J5
            # print(abs((J_new_PIE07 - J_PIE07) / J_new_PIE07))
            # print(abs((J_new_PIE09 - J_PIE09) / J_new_PIE09))
            if (abs((J_new_PIE07 - J_PIE07) / J_new_PIE07)
                    + abs((J_new_PIE09 - J_PIE09) / J_new_PIE09) < epsilon):
                break
        print(m)

        print('PIE07域')
        Y_07 = (W_PIE07.T * data_PIE07).T
        raw_PIE07, column_PIE07 = Y_07.shape
        PIE07_pre = []
        for i in range(raw_PIE07):
            _positon = np.argmax(Y_07[i])
            m, n = divmod(_positon, column_PIE07)
            PIE07_pre.append(n + 1)
        # print(PIE07_pre)
        NMI_PIE07 = metrics.normalized_mutual_info_score(label_PIE07, PIE07_pre)
        RI_PIE07 = RandIndex.rand_index_score(label_PIE07, PIE07_pre)
        #print('NMI大小：', round(NMI_PIE07, 4))
        #print('RI：', round(RI_PIE07, 4))
        NMI_t1_parameter[param_beta, param_gama] = round(NMI_PIE07, 4)
        RI_t1_parameter[param_beta, param_gama] = round(RI_PIE07, 4)

        print('PIE09域')
        Y_09 = (W_PIE09.T * data_PIE09).T
        raw_PIE09, column_PIE09 = Y_09.shape
        PIE09_pre = []
        for i in range(raw_PIE09):
            _positon = np.argmax(Y_09[i])
            m, n = divmod(_positon, column_PIE09)
            PIE09_pre.append(n + 1)
        # print(PIE09_pre)
        NMI_PIE09 = metrics.normalized_mutual_info_score(label_PIE09, PIE09_pre)
        RI_PIE09 = RandIndex.rand_index_score(label_PIE09, PIE09_pre)
        #print('NMI大小：', round(NMI_PIE09, 4))
        #print('RI：', round(RI_PIE09, 4))
        NMI_t2_parameter[param_beta, param_gama] = round(NMI_PIE09, 4)
        RI_t2_parameter[param_beta, param_gama] = round(RI_PIE09, 4)

        print(NMI_t1_parameter)
        print(NMI_t2_parameter)
        scipy.io.savemat('PIE/27/07,09/beta_gama.mat',
                         {'NMI_t1': NMI_t1_parameter, 'RI_t1': RI_t1_parameter,
                          'NMI_t2': NMI_t2_parameter, 'RI_t2': RI_t2_parameter})
