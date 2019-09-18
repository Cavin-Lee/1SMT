# -*- encoding:utf-8 -*-

'''
一个源域：
PIE05：类别分别为1-20
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

classnum_PIE05=20; classnum_PIE07=9; classnum_PIE09=9
M=2   #目标域个数

PIE05=scipy.io.loadmat('PIE/05/PIE05.mat')
PIE07=scipy.io.loadmat('PIE/05/07,09/PIE07.mat')      #第一个目标域

data_PIE05=PIE05['Xs']; data_PIE07=PIE07['Xt'];
label_05=PIE05['Ys']; label_07=PIE07['Yt'];
U_PIE05=np.mat(PIE05['Us'])

'''Dimension reduction'''
pca_05=KernelPCA(n_components=100,kernel='rbf')
pca_07=KernelPCA(n_components=100,kernel='rbf')

data_PIE05=pca_05.fit_transform(data_PIE05)  #大小均为n*d'
data_PIE07=pca_07.fit_transform(data_PIE07)

'''Preprocessing'''
data_PIE05=np.mat(preprocessing.scale(data_PIE05)).T   #大小均为d'*n
data_PIE07=np.mat(preprocessing.scale(data_PIE07)).T

label_PIE05=[]; label_PIE07=[];

for i in range(len(label_05)):
    d=int(label_05[i])
    label_PIE05.append(d)
for i in range(len(label_07)):
    d=int(label_07[i])
    label_PIE07.append(d)


'''WS of source domain'''
Lambda_PIE05=0.5
W_PIE05=source_domain.SLMC_W(data_PIE05,U_PIE05,Lambda_PIE05,classnum_PIE05)

Maxiter = 200; epsilon = 10**(-7); Lambda = 0.5
Y_PIE07 = np.mat(np.eye(classnum_PIE07, dtype=int))

U_PIE07 = np.mat(source_domain.Normization(np.random.random((classnum_PIE07,data_PIE07.shape[1]))))

U1 = U_PIE07;

print("PIE07 Domain:")
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
print('NMI：',round(NMI_1,4))
print('RI：',round(RI_1,4))



'''计算目标域的情况'''
print('目标域的情况')
Lambda_PIE07=1;
beta = [0.1,2 ,4,8,10,12,14,16]
eta = 0.1
r = 10
A = np.mat(np.random.random((classnum_PIE07, classnum_PIE05)))

a,b=A.shape
print(a)
print(b)
K=classnum_PIE07
NMI_t1_parameter = np.mat(np.zeros((8,8)))
RI_t1_parameter = np.mat(np.zeros((8,8)))

for param_beta in range(len(beta)):


        print(' ')
        print('the value of parameter：', beta[param_beta])

        A_tar = A;
        U_PIE07_tar = U_PIE07;


        J_PIE07 = J_new_PIE07 = float("inf")

        for m in range(Maxiter):
            W_PIE07 = target_domain.target_WTj(data_PIE07, U_PIE07_tar, W_PIE05,A,Lambda_PIE07,beta[param_beta])
            U_PIE07_tar = target_domain.target_U(data_PIE07, classnum_PIE07, W_PIE07)
            A = target_domain.Target_A(W_PIE05, W_PIE07, K)
            J_PIE07 = J_new_PIE07

            J2_PIE07 = 0
            for k in range(classnum_PIE07):
                for i in range(data_PIE07.shape[1]):
                    J2_PIE07 = J2_PIE07 + U_PIE07_tar[k, i] ** 2 * np.linalg.norm(
                        (W_PIE07.T * data_PIE07[:, i] - Y_PIE07[:, k]), ord=2) ** 2


            '''目标函数第三,四项的值'''
            J3 = 0

            J3 = J3 + 0.5 * beta[param_beta] * np.linalg.norm(W_PIE05 - W_PIE07 * A, ord=2) ** 2

            classnum_Target = classnum_PIE07


            J_new_PIE07 = 0.5 * (np.linalg.norm(W_PIE07, ord=2) ** 2 + Lambda_PIE07 * J2_PIE07) + J3

            # print(abs((J_new_PIE07 - J_PIE07) / J_new_PIE07))

            if (abs((J_new_PIE07 - J_PIE07) / J_new_PIE07) )< epsilon:
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
        NMI_t1_parameter[param_beta] = round(NMI_PIE07, 4)
        RI_t1_parameter[param_beta] = round(RI_PIE07, 4)



        print(NMI_t1_parameter)

        scipy.io.savemat('PIE/05/07,09/beta_gama05to07.mat',
                         {'NMI_t1': NMI_t1_parameter, 'RI_t1': RI_t1_parameter})