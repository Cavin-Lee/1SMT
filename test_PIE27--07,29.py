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

classnum_PIE27=20; classnum_PIE07=9; classnum_PIE29=9
M=2   #目标域个数

PIE27=scipy.io.loadmat('PIE/27/PIE27.mat')
PIE07=scipy.io.loadmat('PIE/27/07,29/PIE07.mat')      #第一个目标域
PIE29=scipy.io.loadmat('PIE/27/07,29/PIE29.mat')      #第二个目标域

data_PIE27=PIE27['Xs']; data_PIE07=PIE07['Xt']; data_PIE29=PIE29['Xt']
label_27=PIE27['Ys']; label_07=PIE07['Yt']; label_29=PIE29['Yt']
U_PIE27=np.mat(PIE27['Us'])

'''降维'''
pca_27=KernelPCA(n_components=100,kernel='rbf')
pca_07=KernelPCA(n_components=100,kernel='rbf')
pca_29=KernelPCA(n_components=100,kernel='rbf')

data_PIE27=pca_27.fit_transform(data_PIE27)  #大小均为n*d'
data_PIE07=pca_07.fit_transform(data_PIE07)
data_PIE29=pca_29.fit_transform(data_PIE29)

'''预处理'''
data_PIE27=np.mat(preprocessing.scale(data_PIE27)).T   #大小均为d'*n
data_PIE07=np.mat(preprocessing.scale(data_PIE07)).T
data_PIE29=np.mat(preprocessing.scale(data_PIE29)).T

np.savetxt('PIE/27/PIE27(4).csv', data_PIE27, delimiter = ',')
np.savetxt('PIE/27/07,29/PIE29(4).csv', data_PIE29, delimiter = ',')

label_PIE27=[]; label_PIE07=[]; label_PIE29=[]
for i in range(len(label_27)):
    d=int(label_27[i])
    label_PIE27.append(d)
for i in range(len(label_07)):
    d=int(label_07[i])
    label_PIE07.append(d)
for i in range(len(label_29)):
    d=int(label_29[i])
    label_PIE29.append(d)

'''PIE27源域的WS'''
Lambda_PIE27=0.5
W_PIE27=source_domain.SLMC_W(data_PIE27,U_PIE27,Lambda_PIE27,classnum_PIE27)

Maxiter = 200; epsilon = 10**(-4); Lambda = 0.5
Y_PIE07 = np.mat(np.eye(classnum_PIE07, dtype=int))
Y_PIE29 = np.mat(np.eye(classnum_PIE29, dtype=int))
'''
U_PIE07=np.mat(np.loadtxt(open("PIE/05/07,09/PIE09(4).csv","rb"),delimiter=",",skiprows=0))
U_PIE09=np.mat(np.loadtxt(open("PIE/05/07,09/PIE09(4)csv","rb"),delimiter=",",skiprows=0))
'''
U_PIE07 = np.mat(source_domain.Normization(np.random.random((classnum_PIE07,data_PIE07.shape[1]))))
#np.savetxt('PIE/05/07,09/PIE09(4).csv', U_PIE07, delimiter = ',')
U_PIE29=np.mat(source_domain.Normization(np.random.random((classnum_PIE29, data_PIE29.shape[1]))))
#np.savetxt('PIE/05/07,09/PIE09(4)csv', U_PIE09, delimiter = ',')

U1 = U_PIE07; U2=U_PIE29

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

print("PIE29域")
J = J_new = float('inf')
for j in range(Maxiter):

    W2 = source.SLMC_W(data_PIE29, U2, Lambda, classnum_PIE29)
    U2 = source.SLMC_U(data_PIE29, classnum_PIE29, W2)
    J2 = 0
    J = J_new
    for k in range(classnum_PIE29):
        for i in range(data_PIE29.shape[1]):
            J2 = J2 + U2[k, i] ** 2 * np.linalg.norm((W2.T * data_PIE29[:, i] - Y_PIE29[:, k]), ord=2) ** 2
    J_new = 0.5 * np.linalg.norm(W2, ord=2) ** 2 + 0.5 * Lambda * J2
    # print(abs((J_new - J) / J_new))
    if (abs((J_new - J) / J_new) < epsilon):
        break
print(j)
Y2 = (W2.T * data_PIE29).T
raw_2, column_2 = Y2.shape
pre2 = []
for i in range(raw_2):
    _positon = np.argmax(Y2[i])
    m, n = divmod(_positon, column_2)
    pre2.append(n + 1)
NMI_2 = metrics.normalized_mutual_info_score(label_PIE29, pre2)
RI_2 = RandIndex.rand_index_score(label_PIE29, pre2)
print('NMI大小：',round(NMI_2,4))
print('RI：',round(RI_2,4))

'''计算目标域的情况'''
print('目标域的情况')
Lambda_PIE07=1; Lambda_PIE29=1
beta = 50
gama = [10,10]
eta = [0.1]
r = 15
cov_07=[0]*200
cov_29=[0]*200
nmi_07=[0]*200
nmi_29=[0]*200
D = np.mat(np.random.random((data_PIE07.shape[0], r)))  # 随机初始化公共字典
VT = [np.mat(np.random.random((classnum_PIE07, classnum_PIE27))),
      np.mat(np.random.random((classnum_PIE29, classnum_PIE27)))]
V = [np.mat(np.random.random((r, classnum_PIE07))),
     np.mat(np.random.random((r, classnum_PIE29)))]
for param_eta in range(len(eta)):
    print(' ')
    print('the value of eta：', eta[param_eta])

    VT_tar = VT; U_PIE07_tar = U_PIE07; U_PIE29_tar = U_PIE29
    D_tar = D; V_tar = V

    J_PIE07 = J_new_PIE07 = float("inf")
    J_PIE29 = J_new_PIE29 = float("inf")
    for m in range(Maxiter):
        print(m)
        W_PIE07 = target_domain.target_WTj(data_PIE07, U_PIE07_tar, W_PIE27, VT_tar[0], V_tar[0],
                                           D_tar, Lambda_PIE07, gama[0], beta)
        W_PIE29 = target_domain.target_WTj(data_PIE29, U_PIE29_tar, W_PIE27, VT_tar[1], V_tar[1],
                                           D_tar, Lambda_PIE29, gama[1], beta)
        U_PIE07_tar = target_domain.target_U(data_PIE07, classnum_PIE07, W_PIE07)
        U_PIE29_tar = target_domain.target_U(data_PIE29, classnum_PIE29, W_PIE29)
        VT_tar[0] = target_domain.Source_Target_VTj(W_PIE07, W_PIE27, beta, eta[param_eta])
        VT_tar[1] = target_domain.Source_Target_VTj(W_PIE29, W_PIE27, beta, eta[param_eta])
        WT = [W_PIE07, W_PIE29]
        try:
            D_tar = target_domain.Target_D(WT, V_tar, gama, r)
        except Exception as e:
            print(e)
        try:
            D_tar = D_tar / (np.sum(D_tar, axis=0))
        except Exception as e:
            print(e)
        V_tar[0] = target_domain.Target_Vj(W_PIE07, D_tar, gama[0], eta[param_eta])
        V_tar[1] = target_domain.Target_Vj(W_PIE29, D_tar, gama[1], eta[param_eta])
        J_PIE07 = J_new_PIE07
        J_PIE29 = J_new_PIE29
        '''Dslr域的目标函数第二项的值'''
        J2_PIE07 = 0
        for k in range(classnum_PIE07):
            for i in range(data_PIE07.shape[1]):
                J2_PIE07 = J2_PIE07 + U_PIE07_tar[k, i] ** 2 * np.linalg.norm(
                    (W_PIE07.T * data_PIE07[:, i] - Y_PIE07[:, k]), ord=2) ** 2

        '''Webcam域的目标函数第二项的值'''
        J2_PIE29 = 0
        for k in range(classnum_PIE29):
            for i in range(data_PIE29.shape[1]):
                J2_PIE29 = J2_PIE29 + U_PIE29_tar[k, i] ** 2 * np.linalg.norm(
                    (W_PIE29.T * data_PIE29[:, i] - Y_PIE29[:, k]), ord=2) ** 2

        '''目标函数第三,四项的值'''
        J3 = 0
        for j in range(M):
            J3 = J3 + 0.5 * beta * np.linalg.norm(W_PIE27 - WT[j] * VT_tar[j], ord=2) ** 2
            + 0.5 * gama[j] * np.linalg.norm(WT[j] - D_tar * V_tar[j], ord=2) ** 2
        classnum_Target = [classnum_PIE07, classnum_PIE29]
        '''目标函数第五，六项的值'''
        J5 = 0
        for j in range(M):
            V_temp = V_tar[j];
            VT_temp = VT_tar[j]
            for i in range(r):
                J5 = J5 + np.linalg.norm(V_temp[i, :], ord=2)
            for i in range(classnum_Target[j]):
                J5 = J5 + np.linalg.norm(VT_temp[i, :], ord=2)
            J5 = eta[param_eta] * J5

        J_new_PIE07 = 0.5 * (np.linalg.norm(W_PIE07, ord=2) ** 2 + Lambda_PIE07 * J2_PIE07) + J3 + J5
        J_new_PIE29 = 0.5 * (np.linalg.norm(W_PIE29, ord=2) ** 2 + Lambda_PIE29 * J2_PIE29) + J3 + J5
        # print(abs((J_new_PIE07 - J_PIE07) / J_new_PIE07))
        # print(abs((J_new_PIE09 - J_PIE09) / J_new_PIE09))
        cov_07[m]=J_new_PIE07
        cov_29[m]=J_new_PIE29

        Y_07 = (W_PIE07.T * data_PIE07).T
        raw_PIE07, column_PIE07 = Y_07.shape
        PIE07_pre = []
        for i in range(raw_PIE07):
            _positon = np.argmax(Y_07[i])
            mm, n = divmod(_positon, column_PIE07)
            PIE07_pre.append(n + 1)
        # print(PIE07_pre)
        NMI_PIE07 = metrics.normalized_mutual_info_score(label_PIE07, PIE07_pre)
        nmi_07[m]=NMI_PIE07

        Y_29 = (W_PIE29.T * data_PIE29).T
        raw_PIE29, column_PIE29 = Y_29.shape
        PIE29_pre = []
        for i in range(raw_PIE29):
            _positon = np.argmax(Y_29[i])
            mm, n = divmod(_positon, column_PIE29)
            PIE29_pre.append(n + 1)
        # print(PIE09_pre)
        NMI_PIE29 = metrics.normalized_mutual_info_score(label_PIE29, PIE29_pre)
        nmi_29[m]=NMI_PIE29

        if (abs((J_new_PIE07 - J_PIE07) / J_new_PIE07)<epsilon and
                 abs((J_new_PIE29 - J_PIE29) / J_new_PIE29) < epsilon):
            break
print(cov_07)
print(cov_29)
print(nmi_07)
print(nmi_29)
scipy.io.savemat('PIE/27/07,29/cov4.mat',
                         {'cov_07': cov_07, 'cov_29': cov_29,
                          'nmi_07': nmi_07, 'nmi_29': nmi_29})
'''
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
    print('NMI大小：', round(NMI_PIE07, 4))
    print('RI：', round(RI_PIE07, 4))

    print('PIE29域')
    Y_29 = (W_PIE29.T * data_PIE29).T
    raw_PIE29, column_PIE29 = Y_29.shape
    PIE29_pre = []
    for i in range(raw_PIE29):
        _positon = np.argmax(Y_29[i])
        m, n = divmod(_positon, column_PIE29)
        PIE29_pre.append(n + 1)
    # print(PIE09_pre)
    NMI_PIE29 = metrics.normalized_mutual_info_score(label_PIE29, PIE29_pre)
    RI_PIE29 = RandIndex.rand_index_score(label_PIE29, PIE29_pre)
    print('NMI大小：', round(NMI_PIE29, 4))
    print('RI：', round(RI_PIE29, 4))
'''