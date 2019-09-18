# -*- encoding:utf-8 -*-

'''
一个源域：
Amazon：类别分别为1,2,3,4,5,6,7,8,9,10
两个源域（共享四个类）
Dslr：类别分别为1-6
Webcam：类别分别为3-8
值得注意的是：目标域总类是源域的子类
'''

import numpy as np
import scipy.io
import source_domain,target_domain,RandIndex,source,target
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn import preprocessing

classnum_Amazon=10; classnum_Dslr=5; classnum_Webcam=4
M=2   #目标域个数

Amazon=scipy.io.loadmat('CADW/Amazon.mat')
Dslr=scipy.io.loadmat('CADW/Dslr(4).mat')      #第一个目标域
Webcam=scipy.io.loadmat('CADW/Webcam(4).mat')      #第二个目标域

data_Amazon=Amazon['Xs']; data_Dslr=Dslr['Xt']; data_Webcam=Webcam['Xt']
label_A=Amazon['Ys']; label_D=Dslr['Yt']; label_W=Webcam['Yt']
U_Amazon=np.mat(Amazon['Us'])

'''降维'''
pca_A=KernelPCA(n_components=80,kernel='rbf')
pca_D=KernelPCA(n_components=80,kernel='rbf')
pca_W=KernelPCA(n_components=80,kernel='rbf')

data_Webcam=pca_D.fit_transform(data_Webcam)  #大小均为n*d'
data_Amazon=pca_A.fit_transform(data_Amazon)
data_Dslr=pca_D.fit_transform(data_Dslr)

'''预处理'''
data_Webcam=np.mat(preprocessing.scale(data_Webcam)).T   #大小均为d'*n
data_Amazon=np.mat(preprocessing.scale(data_Amazon)).T
data_Dslr=np.mat(preprocessing.scale(data_Dslr)).T

label_Webcam=[]; label_Amazon=[]; label_Dslr=[]
for i in range(len(label_W)):
    d=int(label_W[i])
    label_Webcam.append(d)
for i in range(len(label_D)):
    d=int(label_D[i])
    label_Dslr.append(d)
for i in range(len(label_A)):
    d=int(label_A[i])
    label_Amazon.append(d)

'''Amazon源域的WS'''
Lambda_Amazon=0.5
W_Amazon=source_domain.SLMC_W(data_Amazon,U_Amazon,Lambda_Amazon,classnum_Amazon)

Maxiter = 200; epsilon = 10**(-4); Lambda = 0.5
Y_Dslr = np.mat(np.eye(classnum_Dslr, dtype=int))
Y_Webcam=np.mat(np.eye(classnum_Webcam, dtype=int))
U_Dslr = np.mat(source_domain.Normization(np.random.random((classnum_Dslr,data_Dslr.shape[1]))))
U_Webcam=np.mat(source_domain.Normization(np.random.random((classnum_Webcam, data_Webcam.shape[1]))))
U1 = U_Dslr; U2=U_Webcam

print("Dslr域")
J = J_new = float('inf')
for j in range(Maxiter):
    W1 = source.SLMC_W(data_Dslr, U1, Lambda, classnum_Dslr)
    U1 = source.SLMC_U(data_Dslr, classnum_Dslr, W1)
    J2 = 0
    J = J_new
    for k in range(classnum_Dslr):
        for i in range(data_Dslr.shape[1]):
            J2 = J2 + U1[k, i] ** 2 * np.linalg.norm((W1.T * data_Dslr[:, i] - Y_Dslr[:, k]), ord=2) ** 2
    J_new = 0.5 * np.linalg.norm(W1, ord=2) ** 2 + 0.5 * Lambda * J2
    # print(abs((J_new - J) / J_new))
    if (abs((J_new - J) / J_new) < epsilon):
        break
print(j)
Y1 = (W1.T * data_Dslr).T
raw_1, column_1 = Y1.shape
pre1 = []
for i in range(raw_1):
    _positon = np.argmax(Y1[i])
    m, n = divmod(_positon, column_1)
    pre1.append(n + 1)
#print(pre1)
NMI_1 = metrics.normalized_mutual_info_score(label_Dslr, pre1)
RI_1 = RandIndex.rand_index_score(label_Dslr, pre1)
print('NMI大小：',round(NMI_1,4))
print('RI：',round(RI_1,4))

print("Webcam域")
J = J_new = float('inf')
for j in range(Maxiter):
    W2 = source.SLMC_W(data_Webcam, U2, Lambda, classnum_Webcam)
    U2 = source.SLMC_U(data_Webcam, classnum_Webcam, W2)
    J2 = 0
    J = J_new
    for k in range(classnum_Webcam):
        for i in range(data_Webcam.shape[1]):
            J2 = J2 + U2[k, i] ** 2 * np.linalg.norm((W2.T * data_Webcam[:, i] - Y_Webcam[:, k]), ord=2) ** 2
    J_new = 0.5 * np.linalg.norm(W2, ord=2) ** 2 + 0.5 * Lambda * J2
    # print(abs((J_new - J) / J_new))
    if (abs((J_new - J) / J_new) < epsilon):
        break
print(j)
Y2 = (W2.T * data_Webcam).T
raw_2, column_2 = Y2.shape
pre2 = []
for i in range(raw_2):
    _positon = np.argmax(Y2[i])
    m, n = divmod(_positon, column_2)
    pre2.append(n + 1)
#print(pre2)
NMI_2 = metrics.normalized_mutual_info_score(label_Webcam, pre2)
RI_2 = RandIndex.rand_index_score(label_Webcam, pre2)
print('NMI大小：',round(NMI_2,4))
print('RI：',round(RI_2,4))

'''计算目标域的情况'''
print('目标域的情况')
Lambda_Dslr=1; Lambda_Webcam=1
beta = 100
gama = 50
eta =[0.00001,0.0001,0.001,0.01,0.1,1,10]
r = 10

NMI_D_parameter = np.mat(np.zeros((7,1)))
NMI_W_parameter = np.mat(np.zeros((7,1)))
RI_D_parameter = np.mat(np.zeros((7,1)))
RI_W_parameter = np.mat(np.zeros((7,1)))
VT = [np.mat(np.random.random((classnum_Dslr, classnum_Amazon))),
      np.mat(np.random.random((classnum_Webcam, classnum_Amazon)))]
D = np.mat(np.random.random((data_Dslr.shape[0], r)))  # 随机初始化公共字典
V = [np.mat(np.random.random((r, classnum_Dslr))),
     np.mat(np.random.random((r, classnum_Webcam)))]
for param_eta in range(len(eta)):
    print(' ')
    print('eta参数的值：', eta[param_eta])
    gama_tar = [gama, gama]

    VT_tar = VT;
    U_Dslr_tar = U_Dslr;
    U_Webcam_tar = U_Webcam
    D_tar = D;
    V_tar = V

    J_Dslr = J_new_Dslr = float("inf")
    J_Webcam = J_new_Webcam = float("inf")
    for m in range(Maxiter):

        W_Dslr = target_domain.target_WTj(data_Dslr, U_Dslr_tar, W_Amazon, VT_tar[0], V_tar[0],
                                          D_tar, Lambda_Dslr, gama_tar[0], beta)
        W_Webcam = target_domain.target_WTj(data_Webcam, U_Webcam_tar, W_Amazon, VT_tar[1], V_tar[1],
                                            D_tar, Lambda_Webcam, gama_tar[1], beta)
        U_Dslr_tar = target_domain.target_U(data_Dslr, classnum_Dslr, W_Dslr)
        U_Webcam_tar = target_domain.target_U(data_Webcam, classnum_Webcam, W_Webcam)
        VT_tar[0] = target_domain.Source_Target_VTj(W_Dslr, W_Amazon, beta, eta[param_eta])
        VT_tar[1] = target_domain.Source_Target_VTj(W_Webcam, W_Amazon, beta, eta[param_eta])
        WT = [W_Dslr, W_Webcam]
        try:
            D_tar = target_domain.Target_D(WT, V_tar, gama_tar, r)
        except Exception as e:
            print(e)
        try:
            D_tar = D_tar / (np.sum(D_tar, axis=0))
        except Exception as e:
            print(e)
            break
        V_tar[0] = target_domain.Target_Vj(W_Dslr, D_tar, gama_tar[0], eta[param_eta])
        V_tar[1] = target_domain.Target_Vj(W_Webcam, D_tar, gama_tar[1], eta[param_eta])
        J_Dslr = J_new_Dslr
        J_Webcam = J_new_Webcam
        '''Dslr域的目标函数第二项的值'''
        J2_Dslr = 0
        for k in range(classnum_Dslr):
            for i in range(data_Dslr.shape[1]):
                J2_Dslr = J2_Dslr + U_Dslr_tar[k, i] ** 2 * np.linalg.norm(
                    (W_Dslr.T * data_Dslr[:, i] - Y_Dslr[:, k]),
                    ord=2) ** 2

        '''Webcam域的目标函数第二项的值'''
        J2_Webcam = 0
        for k in range(classnum_Webcam):
            for i in range(data_Webcam.shape[1]):
                J2_Webcam = J2_Webcam + U_Webcam_tar[k, i] ** 2 * np.linalg.norm(
                    (W_Webcam.T * data_Webcam[:, i] - Y_Webcam[:, k]), ord=2) ** 2

        '''目标函数第三,四项的值'''
        J3 = 0
        for j in range(M):
            J3 = J3 + 0.5 * beta * np.linalg.norm(W_Amazon - WT[j] * VT_tar[j], ord=2) ** 2
            + 0.5 * gama_tar[j] * np.linalg.norm(WT[j] - D_tar * V_tar[j], ord=2) ** 2
        classnum_Target = [classnum_Dslr, classnum_Webcam]
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

        J_new_Dslr = 0.5 * (np.linalg.norm(W_Dslr, ord=2) ** 2 + Lambda_Dslr * J2_Dslr) + J3 + J5
        J_new_Webcam = 0.5 * (np.linalg.norm(W_Webcam, ord=2) ** 2 + Lambda_Webcam * J2_Webcam) + J3 + J5
        # print(abs((J_new_Dslr - J_Dslr) / J_new_Dslr))
        # print(abs((J_new_Webcam - J_Webcam) / J_new_Webcam))
        if (abs((J_new_Dslr - J_Dslr) / J_new_Dslr) < epsilon and abs(
                    (J_new_Webcam - J_Webcam) / J_new_Webcam) < epsilon):
            break
    print(m)
    print('Dslr域')
    Y_D = (W_Dslr.T * data_Dslr).T
    raw_D, column_D = Y_D.shape
    Dslr_pre = []
    for i in range(raw_D):
        _positon = np.argmax(Y_D[i])
        m, n = divmod(_positon, column_D)
        Dslr_pre.append(n + 1)
    # print(Dslr_pre)
    NMI_D = metrics.normalized_mutual_info_score(label_Dslr, Dslr_pre)
    RI_D = RandIndex.rand_index_score(label_Dslr, Dslr_pre)
    NMI_D_parameter[param_eta,0] = round(NMI_D, 4)
    RI_D_parameter[param_eta, 0] = round(RI_D, 4)
    # print('NMI大小：', round(NMI_D, 4))
    # print('RI：', round(RI_D, 4))

    print('Webcam域')
    Y_W = (W_Webcam.T * data_Webcam).T
    raw_W, column_W = Y_W.shape
    Webcam_pre = []
    for i in range(raw_W):
        _positon = np.argmax(Y_W[i])
        m, n = divmod(_positon, column_W)
        Webcam_pre.append(n + 1)
    # print(Webcam_pre)
    NMI_W = metrics.normalized_mutual_info_score(label_Webcam, Webcam_pre)
    RI_W = RandIndex.rand_index_score(label_Webcam, Webcam_pre)
    NMI_W_parameter[param_eta, 0] = round(NMI_W, 4)
    RI_W_parameter[param_eta, 0] = round(RI_W, 4)
    # print('NMI大小：', round(NMI_W, 4))
    # print('RI：', round(RI_W, 4))
    print(NMI_D_parameter)
    print(RI_D_parameter)
    print(NMI_W_parameter)
    print(RI_W_parameter)
    scipy.io.savemat('CADW/ADW_eta(NMI,RI).mat', {'NMI_D': NMI_D_parameter, 'RI_D': RI_D_parameter,
                                              'NMI_W': NMI_W_parameter, 'RI_W': RI_W_parameter})

