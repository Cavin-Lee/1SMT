# -*- encoding:utf-8 -*-

'''
one source：
Y2：38
two targets:
Y5: 18      class:1-18
Y3：18      class:7-24
Y4: 18      class:7-18,25-30
Y1: 18      class:7-18,31-36
'''

import numpy as np
import scipy.io
import source_domain,target_domain,RandIndex,source,target
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn import preprocessing

classnum_source=38; classnum_target1=18; classnum_target2=18; classnum_target3=18; classnum_target4=18
M=4   #the number of target domain
'''
Source=scipy.io.loadmat('YaleB/4/Y2--Y5,Y3,Y4,Y1/Y2.mat')
Target1=scipy.io.loadmat('YaleB/4/Y2--Y5,Y3,Y4,Y1/Y5.mat')      # first target
Target2 =scipy.io.loadmat('YaleB/4/Y2--Y5,Y3,Y4,Y1/Y3.mat')      # second target
Target3 =scipy.io.loadmat('YaleB/4/Y2--Y5,Y3,Y4,Y1/Y4.mat')      # third target
Target4 =scipy.io.loadmat('YaleB/4/Y2--Y5,Y3,Y4,Y1/Y1.mat')      # forth target


Source=scipy.io.loadmat('YaleB/4/Y3--Y5,Y2,Y4,Y1/Y3.mat')
Target1=scipy.io.loadmat('YaleB/4/Y3--Y5,Y2,Y4,Y1/Y5.mat')      # first target
Target2 =scipy.io.loadmat('YaleB/4/Y3--Y5,Y2,Y4,Y1/Y2.mat')      # second target
Target3 =scipy.io.loadmat('YaleB/4/Y3--Y5,Y2,Y4,Y1/Y4.mat')      # third target
Target4 =scipy.io.loadmat('YaleB/4/Y3--Y5,Y2,Y4,Y1/Y1.mat')      # forth target

Source=scipy.io.loadmat('YaleB/4/Y4--Y5,Y2,Y3,Y1/Y4.mat')
Target1=scipy.io.loadmat('YaleB/4/Y4--Y5,Y2,Y3,Y1/Y5.mat')      # first target
Target2 =scipy.io.loadmat('YaleB/4/Y4--Y5,Y2,Y3,Y1/Y2.mat')      # second target
Target3 =scipy.io.loadmat('YaleB/4/Y4--Y5,Y2,Y3,Y1/Y3.mat')      # third target
Target4 =scipy.io.loadmat('YaleB/4/Y4--Y5,Y2,Y3,Y1/Y1.mat')      # forth target
'''
Source=scipy.io.loadmat('YaleB/4/Y5--Y2,Y3,Y4,Y1/Y5.mat')
Target1=scipy.io.loadmat('YaleB/4/Y5--Y2,Y3,Y4,Y1/Y2.mat')      # first target
Target2 =scipy.io.loadmat('YaleB/4/Y5--Y2,Y3,Y4,Y1/Y3.mat')      # second target
Target3 =scipy.io.loadmat('YaleB/4/Y5--Y2,Y3,Y4,Y1/Y4.mat')      # third target
Target4 =scipy.io.loadmat('YaleB/4/Y5--Y2,Y3,Y4,Y1/Y1.mat')      # forth target

''''''


data_source=Source['Xs']
data_target1=Target1['Xt']; data_target2=Target2['Xt']; data_target3=Target3['Xt']; data_target4=Target4['Xt']
label_s=Source['Ys']
label_t1=Target1['Yt']; label_t2=Target2['Yt']; label_t3=Target3['Yt']; label_t4=Target4['Yt']
U_Source=np.mat(Source['Us'])

'''Dimensionality Reduction'''
pca_s=KernelPCA(n_components=100,kernel='rbf')
pca_t1=KernelPCA(n_components=100,kernel='rbf')
pca_t2=KernelPCA(n_components=100,kernel='rbf')
pca_t3=KernelPCA(n_components=100,kernel='rbf')
pca_t4=KernelPCA(n_components=100,kernel='rbf')

data_source=pca_s.fit_transform(data_source) #大小均为n*d'
data_target1=pca_t1.fit_transform(data_target1)
data_target2=pca_t2.fit_transform(data_target2)
data_target3=pca_t3.fit_transform(data_target3)
data_target4=pca_t4.fit_transform(data_target4)


'''Preprocessing
data_source=np.mat(preprocessing.scale(data_source)).T   #大小均为d'*n
data_target1=np.mat(preprocessing.scale(data_target1)).T
data_target2=np.mat(preprocessing.scale(data_target2)).T
'''
data_source=np.mat(preprocessing.StandardScaler().fit_transform(data_source)).T
data_target1=np.mat(preprocessing.StandardScaler().fit_transform(data_target1)).T
data_target2=np.mat(preprocessing.StandardScaler().fit_transform(data_target2)).T
data_target3=np.mat(preprocessing.StandardScaler().fit_transform(data_target3)).T
data_target4=np.mat(preprocessing.StandardScaler().fit_transform(data_target4)).T

label_source=[]; label_target1=[]; label_target2=[]; label_target3=[]; label_target4=[];
for i in range(len(label_s)):
    d=int(label_s[i])
    label_source.append(d)
for i in range(len(label_t1)):
    d=int(label_t1[i])
    label_target1.append(d)
for i in range(len(label_t2)):
    d=int(label_t2[i])
    label_target2.append(d)
for i in range(len(label_t3)):
    d=int(label_t3[i])
    label_target3.append(d)
for i in range(len(label_t4)):
    d=int(label_t4[i])
    label_target4.append(d)

'''Imagenet:WS'''
Lambda_source=0.5
W_source=source_domain.SLMC_W(data_source,U_Source,Lambda_source,classnum_source)

Maxiter = 200; epsilon = 10**(-7); Lambda = 0.5
Y_target1 = np.mat(np.eye(classnum_target1, dtype=int))
Y_target2=np.mat(np.eye(classnum_target2, dtype=int))
Y_target3=np.mat(np.eye(classnum_target3, dtype=int))
Y_target4=np.mat(np.eye(classnum_target4, dtype=int))
U_target1 = np.mat(source_domain.Normization(np.random.random((classnum_target1,data_target1.shape[1]))))
U_target2=np.mat(source_domain.Normization(np.random.random((classnum_target2, data_target2.shape[1]))))
U_target3=np.mat(source_domain.Normization(np.random.random((classnum_target3, data_target3.shape[1]))))
U_target4=np.mat(source_domain.Normization(np.random.random((classnum_target4, data_target4.shape[1]))))
U1 = U_target1; U2=U_target2; U3=U_target3; U4=U_target4

'''Target Domain'''
print('-------------------Target Domain-------------------')
Lambda_target1=1; Lambda_target2=1; Lambda_target3=1; Lambda_target4=1
beta = [0.0001,0.001,0.01,0.1,1,10,50,100]
gama = [0.0001,0.001,0.01,0.1,1,10,50,100]
eta =0.01
r = 20

VT = [np.mat(np.random.random((classnum_target1, classnum_source))),
      np.mat(np.random.random((classnum_target2, classnum_source))),
      np.mat(np.random.random((classnum_target3, classnum_source))),
      np.mat(np.random.random((classnum_target4, classnum_source)))]
D = np.mat(np.random.random((data_target1.shape[0], r)))  # 随机初始化公共字典
V = [np.mat(np.random.random((r, classnum_target1))),
     np.mat(np.random.random((r, classnum_target2))),
     np.mat(np.random.random((r, classnum_target3))),
     np.mat(np.random.random((r, classnum_target4)))]

NMI_Y1_parameter = np.mat(np.zeros((8,8)))
NMI_Y3_parameter = np.mat(np.zeros((8,8)))
NMI_Y4_parameter = np.mat(np.zeros((8,8)))
NMI_Y5_parameter = np.mat(np.zeros((8,8)))
RI_Y1_parameter = np.mat(np.zeros((8,8)))
RI_Y3_parameter = np.mat(np.zeros((8,8)))
RI_Y4_parameter = np.mat(np.zeros((8,8)))
RI_Y5_parameter = np.mat(np.zeros((8,8)))

for param_beta in range(len(beta)):
    for param_gama in range(len(gama)):
        gama_tar = [gama[param_gama], gama[param_gama], gama[param_gama], gama[param_gama]]
        print(' ')
        print('the value of parameter：', beta[param_beta], gama[param_gama])
        # gama_tar=[gama[param_eta],gama[param_eta]]

        VT_tar = VT;
        U_target1_tar = U_target1;
        U_target2_tar = U_target2;
        U_target3_tar = U_target3;
        U_target4_tar = U_target4;
        D_tar = D;
        V_tar = V

        J_target1 = J_new_target1 = float("inf")
        J_target2 = J_new_target2 = float("inf")
        J_target3 = J_new_target3 = float("inf")
        J_target4 = J_new_target4 = float("inf")

        for m in range(Maxiter):
            W_target1 = target_domain.target_WTj(data_target1, U_target1_tar, W_source, VT_tar[0], V_tar[0],
                                                 D_tar, Lambda_target1, gama_tar[0], beta[param_beta])
            W_target2 = target_domain.target_WTj(data_target2, U_target2_tar, W_source, VT_tar[1], V_tar[1],
                                                 D_tar, Lambda_target2, gama_tar[1], beta[param_beta])
            W_target3 = target_domain.target_WTj(data_target3, U_target3_tar, W_source, VT_tar[2], V_tar[2],
                                                 D_tar, Lambda_target3, gama_tar[2], beta[param_beta])
            W_target4 = target_domain.target_WTj(data_target4, U_target4_tar, W_source, VT_tar[3], V_tar[3],
                                                 D_tar, Lambda_target4, gama_tar[3], beta[param_beta])
            U_target1_tar = target_domain.target_U(data_target1, classnum_target1, W_target1)
            U_target2_tar = target_domain.target_U(data_target2, classnum_target2, W_target2)
            U_target3_tar = target_domain.target_U(data_target3, classnum_target3, W_target3)
            U_target4_tar = target_domain.target_U(data_target4, classnum_target4, W_target4)
            VT_tar[0] = target_domain.Source_Target_VTj(W_target1, W_source, beta[param_beta], eta)
            VT_tar[1] = target_domain.Source_Target_VTj(W_target2, W_source, beta[param_beta], eta)
            VT_tar[2] = target_domain.Source_Target_VTj(W_target3, W_source, beta[param_beta], eta)
            VT_tar[3] = target_domain.Source_Target_VTj(W_target4, W_source, beta[param_beta], eta)
            WT = [W_target1, W_target2, W_target3, W_target4]
            try:
                D_tar = target_domain.Target_D(WT, V_tar, gama_tar, r)
            except Exception as e:
                print(e)
            try:
                D_tar = D_tar / (np.sum(D_tar, axis=0))
            except Exception as e:
                print(e)
            V_tar[0] = target_domain.Target_Vj(W_target1, D_tar, gama_tar[0], eta)
            V_tar[1] = target_domain.Target_Vj(W_target2, D_tar, gama_tar[1], eta)
            V_tar[2] = target_domain.Target_Vj(W_target3, D_tar, gama_tar[2], eta)
            V_tar[3] = target_domain.Target_Vj(W_target4, D_tar, gama_tar[3], eta)
            J_target1 = J_new_target1
            J_target2 = J_new_target2
            J_target3 = J_new_target3
            J_target4 = J_new_target4

            J2_target1 = 0
            for k in range(classnum_target1):
                for i in range(data_target1.shape[1]):
                    J2_target1 = J2_target1 + U_target1_tar[k, i] ** 2 * np.linalg.norm(
                        (W_target1.T * data_target1[:, i] - Y_target1[:, k]),
                        ord=2) ** 2
            J2_target2 = 0
            for k in range(classnum_target2):
                for i in range(data_target2.shape[1]):
                    J2_target2 = J2_target2 + U_target2_tar[k, i] ** 2 * np.linalg.norm(
                        (W_target2.T * data_target2[:, i] - Y_target2[:, k]), ord=2) ** 2

            J2_target3 = 0
            for k in range(classnum_target3):
                for i in range(data_target3.shape[1]):
                    J2_target3 = J2_target3 + U_target3_tar[k, i] ** 2 * np.linalg.norm(
                        (W_target3.T * data_target3[:, i] - Y_target3[:, k]), ord=2) ** 2

            J2_target4 = 0
            for k in range(classnum_target4):
                for i in range(data_target4.shape[1]):
                    J2_target4 = J2_target4 + U_target4_tar[k, i] ** 2 * np.linalg.norm(
                        (W_target4.T * data_target4[:, i] - Y_target4[:, k]), ord=2) ** 2

            J3 = 0
            for j in range(M):
                J3 = J3 + 0.5 * beta[param_beta] * np.linalg.norm(W_source - WT[j] * VT_tar[j], ord=2) ** 2
                + 0.5 * gama_tar[j] * np.linalg.norm(WT[j] - D_tar * V_tar[j], ord=2) ** 2
            classnum_Target = [classnum_target1, classnum_target2, classnum_target3, classnum_target4]

            J5 = 0
            for j in range(M):
                V_temp = V_tar[j];
                VT_temp = VT_tar[j]
                for i in range(r):
                    J5 = J5 + np.linalg.norm(V_temp[i, :], ord=2)
                for i in range(classnum_Target[j]):
                    J5 = J5 + np.linalg.norm(VT_temp[i, :], ord=2)
                J5 = eta * J5

            J_new_target1 = 0.5 * (np.linalg.norm(W_target1, ord=2) ** 2 + Lambda_target1 * J2_target1) + J3 + J5
            J_new_target2 = 0.5 * (np.linalg.norm(W_target2, ord=2) ** 2 + Lambda_target2 * J2_target2) + J3 + J5
            J_new_target3 = 0.5 * (np.linalg.norm(W_target3, ord=2) ** 2 + Lambda_target3 * J2_target3) + J3 + J5
            J_new_target4 = 0.5 * (np.linalg.norm(W_target4, ord=2) ** 2 + Lambda_target4 * J2_target4) + J3 + J5
            # print(abs((J_new_Dslr - J_Dslr) / J_new_Dslr))
            # print(abs((J_new_Webcam - J_Webcam) / J_new_Webcam))
            if (abs((J_new_target1 - J_target1) / J_new_target1) < epsilon and
                        abs((J_new_target2 - J_target2) / J_new_target2) < epsilon and
                        abs((J_new_target3 - J_target3) / J_new_target3) < epsilon and
                        abs((J_new_target4 - J_target4) / J_new_target4) < epsilon):
                break
        print(m)
        print('--------------------------Y5 Domain--------------------------')
        Y_t1 = (W_target1.T * data_target1).T
        raw_t1, column_t1 = Y_t1.shape
        target1_pre = []
        for i in range(raw_t1):
            _positon = np.argmax(Y_t1[i])
            m, n = divmod(_positon, column_t1)
            target1_pre.append(n + 1)

        NMI_t1 = metrics.normalized_mutual_info_score(label_target1, target1_pre)
        RI_t1 = RandIndex.rand_index_score(label_target1, target1_pre)
        #print('NMI：', round(NMI_t1, 4))
        #print('RI：', round(RI_t1, 4))
        NMI_Y1_parameter[param_beta, param_gama] = round(NMI_t1, 4)
        RI_Y1_parameter[param_beta, param_gama] = round(RI_t1, 4)


        print('--------------------------Y3 Domain--------------------------')
        Y_t2 = (W_target2.T * data_target2).T
        raw_t2, column_t2 = Y_t2.shape
        target2_pre = []
        for i in range(raw_t2):
            _positon = np.argmax(Y_t2[i])
            m, n = divmod(_positon, column_t2)
            target2_pre.append(n + 1)
        NMI_t2 = metrics.normalized_mutual_info_score(label_target2, target2_pre)
        RI_t2 = RandIndex.rand_index_score(label_target2, target2_pre)
        #print('NMI：', round(NMI_t2, 4))
        #print('RI：', round(RI_t2, 4))
        NMI_Y3_parameter[param_beta, param_gama] = round(NMI_t2, 4)
        RI_Y3_parameter[param_beta, param_gama] = round(RI_t2, 4)

        print('--------------------------Y4 Domain--------------------------')
        Y_t3 = (W_target3.T * data_target3).T
        raw_t3, column_t3 = Y_t3.shape
        target3_pre = []
        for i in range(raw_t3):
            _positon = np.argmax(Y_t3[i])
            m, n = divmod(_positon, column_t3)
            target3_pre.append(n + 1)
        NMI_t3 = metrics.normalized_mutual_info_score(label_target3, target3_pre)
        RI_t3 = RandIndex.rand_index_score(label_target3, target3_pre)
        #print('NMI：', round(NMI_t3, 4))
        #print('RI：', round(RI_t3, 4))
        NMI_Y4_parameter[param_beta, param_gama] = round(NMI_t3, 4)
        RI_Y4_parameter[param_beta, param_gama] = round(RI_t3, 4)

        print('--------------------------Y1 Domain--------------------------')
        Y_t4 = (W_target4.T * data_target4).T
        raw_t4, column_t4 = Y_t4.shape
        target4_pre = []
        for i in range(raw_t4):
            _positon = np.argmax(Y_t4[i])
            m, n = divmod(_positon, column_t4)
            target4_pre.append(n + 1)
        NMI_t4 = metrics.normalized_mutual_info_score(label_target4, target4_pre)
        RI_t4 = RandIndex.rand_index_score(label_target4, target4_pre)
        #print('NMI：', round(NMI_t4, 4))
        #print('RI：', round(RI_t4, 4))
        NMI_Y5_parameter[param_beta, param_gama] = round(NMI_t4, 4)
        RI_Y5_parameter[param_beta, param_gama] = round(RI_t4, 4)
        print(NMI_Y1_parameter)
        print(NMI_Y3_parameter)
        print(NMI_Y4_parameter)
        print(NMI_Y5_parameter)
        scipy.io.savemat('YaleB/4/beta_gama.mat',
                         {'NMI_t1': NMI_Y1_parameter, 'RI_t1': RI_Y1_parameter,
                          'NMI_t2': NMI_Y3_parameter, 'RI_t2': RI_Y3_parameter,
                          'NMI_t3': NMI_Y4_parameter, 'RI_t3': RI_Y4_parameter,
                          'NMI_t4': NMI_Y5_parameter, 'RI_t4': RI_Y5_parameter})
