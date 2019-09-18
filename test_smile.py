

'''
源域：
smile：类别分别为1-3
目标域
smile_target：类别分别为1-3

'''

import numpy as np
import scipy.io
import source_domain,target_domain,RandIndex,source,target
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn import preprocessing

classnum_smile_source=3; classnum_smile_target=3;

smile_source=scipy.io.loadmat('smile/smile_source.mat')
smile_target=scipy.io.loadmat('smile/smile_target_100.mat')      #第一个目标域

data_smile_source=smile_source['Xs']; data_smile_target=smile_target['Xt'];
label_source=smile_source['Ys']; label_target=smile_target['Yt'];
U_smile_source=np.mat(smile_source['Us'])


'''Preprocessing'''
data_smile_source=np.mat(preprocessing.scale(data_smile_source)).T   #大小均为d'*n
data_smile_target=np.mat(preprocessing.scale(data_smile_target)).T

label_smile_source=[]; label_smile_target=[];

for i in range(len(label_source)):
    d=int(label_source[i])
    label_smile_source.append(d)
for i in range(len(label_target)):
    d=int(label_target[i])
    label_smile_target.append(d)

'''WS of source domain'''
Lambda_smile_source=0.5
W_smile_source=source_domain.SLMC_W(data_smile_source,U_smile_source,Lambda_smile_source,classnum_smile_source)
print(W_smile_source)
Maxiter = 200; epsilon = 10**(-7); Lambda = 0.5
Y_smile_target = np.mat(np.eye(classnum_smile_target, dtype=int))
print(Y_smile_target)
U_smile_target = np.mat(source_domain.Normization(np.random.random((classnum_smile_target,data_smile_target.shape[1]))))

U1 = U_smile_target;

print("smile_target Domain:")
J = J_new = float('inf')
for j in range(Maxiter):
    W1 = source.SLMC_W(data_smile_target, U1, Lambda, classnum_smile_target)

    U1 = source.SLMC_U(data_smile_target, classnum_smile_target, W1)

    J2 = 0
    J = J_new
    for k in range(classnum_smile_target):
        for i in range(data_smile_target.shape[1]):


            J2 = J2 + U1[k, i] ** 2 * np.linalg.norm((W1.T * data_smile_target[:, i] - Y_smile_target[:, k]), ord=2) ** 2
    J_new = 0.5 * np.linalg.norm(W1, ord=2) ** 2 + 0.5 * Lambda * J2
    # print(abs((J_new - J) / J_new))
    if (abs((J_new - J) / J_new) < epsilon):
        break
print(j)
Y1 = (W1.T * data_smile_target).T
raw_1, column_1 = Y1.shape
pre1 = []
for i in range(raw_1):
    _positon = np.argmax(Y1[i])
    m, n = divmod(_positon, column_1)
    pre1.append(n + 1)


print(label_smile_target)
NMI_1 = metrics.normalized_mutual_info_score(label_smile_target, pre1)
RI_1 = RandIndex.rand_index_score(label_smile_target, pre1)
print('NMI：',round(NMI_1,4))
print('RI：',round(RI_1,4))



'''计算目标域的情况'''
print('目标域的情况')
Lambda_smile_target=0.5;
beta =[0.1,1,10,15,50,75,100,200]
eta = 0.1
r = 10
A = np.mat(np.random.random((classnum_smile_target, classnum_smile_source)))

a,b=A.shape
print(a)
print(b)
K=3
NMI_t1_parameter = np.mat(np.zeros((8,8)))
RI_t1_parameter = np.mat(np.zeros((8,8)))

for param_beta in range(len(beta)):


        print(' ')
        print('the value of parameter：', beta[param_beta])

        A_tar = A;
        U_smile_target_tar = U_smile_target;


        J_smile_target = J_new_smile_target = float("inf")

        for m in range(Maxiter):
            W_smile_target = target_domain.target_WTj(data_smile_target, U_smile_target_tar, W_smile_source,A_tar,Lambda_smile_target,beta[param_beta])
            U_smile_target_tar = target_domain.target_U(data_smile_target, classnum_smile_target, W_smile_target)
            A = target_domain.Target_A(W_smile_source, W_smile_target, K)
            J_smile_target = J_new_smile_target

            J2_smile_target = 0
            for k in range(classnum_smile_target):
                for i in range(data_smile_target.shape[1]):
                    J2_smile_target = J2_smile_target + U_smile_target_tar[k, i] ** 2 * np.linalg.norm(
                        (W_smile_target.T * data_smile_target[:, i] - Y_smile_target[:, k]), ord=2) ** 2


            '''目标函数第三,四项的值'''
            J3 = 0

            J3 = J3 + 0.5 * beta[param_beta] * np.linalg.norm(W_smile_source - W_smile_target * A, ord=2) ** 2

            classnum_Target = classnum_smile_target


            J_new_smile_target = 0.5 * (np.linalg.norm(W_smile_target, ord=2) ** 2 + Lambda_smile_target * J2_smile_target) + J3

            # print(abs((J_new_smile_target - J_smile_target) / J_new_smile_target))

            if (abs((J_new_smile_target - J_smile_target) / J_new_smile_target) )< epsilon:
                break
        print(m)
        print('smile_target域')
        Y_07 = (W_smile_target.T * data_smile_target).T
        raw_smile_target, column_smile_target = Y_07.shape
        smile_target_pre = []
        for i in range(raw_smile_target):
            _positon = np.argmax(Y_07[i])
            m, n = divmod(_positon, column_smile_target)
            smile_target_pre.append(n + 1)
        # print(smile_target_pre)
        NMI_smile_target = metrics.normalized_mutual_info_score(label_smile_target, smile_target_pre)
        RI_smile_target = RandIndex.rand_index_score(label_smile_target, smile_target_pre)
        #print('NMI大小：', round(NMI_smile_target, 4))
        #print('RI：', round(RI_smile_target, 4))
        NMI_t1_parameter[param_beta] = round(NMI_smile_target, 4)
        RI_t1_parameter[param_beta] = round(RI_smile_target, 4)



        print(NMI_t1_parameter)

        scipy.io.savemat('smile/beta_gama05to07.mat',
                         {'NMI_t1': NMI_t1_parameter, 'RI_t1': RI_t1_parameter})