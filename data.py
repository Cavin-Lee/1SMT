# -*- encoding:utf-8 -*-

'''
一个源域：
Amazon：类别分别为1,2,3,4,5,6,7,8,9,10
两个源域（共享四个类）
Dslr：类别分别为1-6
Webcam：类别分别为5-10
值得注意的是：目标域总类是源域的子类
'''

import numpy as np
import scipy.io
import source_domain,target_domain,RandIndex,source,target
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn import preprocessing

classnum_Amazon=10; classnum_Dslr=6; classnum_Webcam=6
M=2   #目标域个数

Amazon=scipy.io.loadmat('CADW/A--DW(2)/Amazon.mat')
Dslr=scipy.io.loadmat('CADW/A--DW(2)/Dslr.mat')      #第一个目标域
Webcam=scipy.io.loadmat('CADW/A--DW(2)/Webcam.mat')      #第二个目标域
Caltech=scipy.io.loadmat('CADW/A--DW(2)/Caltech.mat')      #第二个目标域

data_Amazon=Amazon['Xs']; data_Dslr=Dslr['Xs'].T; data_Webcam=Webcam['Xs'].T;data_Caltech=Caltech['Xs'].T
label_A=Amazon['Ys']; label_D=Dslr['Ys']; label_W=Webcam['Ys']; label_C=Caltech['Ys']
U_Amazon=np.mat(Amazon['Us'])
U_Dslr=np.mat(Dslr['Us'])
U_Webcam=np.mat(Webcam['Us'])
U_Caltech=np.mat(Caltech['Us'])
'''降维'''
pca_A=KernelPCA(n_components=80,kernel='rbf')
pca_D=KernelPCA(n_components=80,kernel='rbf')
pca_W=KernelPCA(n_components=80,kernel='rbf')
pca_C=KernelPCA(n_components=80,kernel='rbf')

data_Webcam=pca_D.fit_transform(data_Webcam)  #大小均为n*d'
data_Amazon=pca_A.fit_transform(data_Amazon)
data_Dslr=pca_D.fit_transform(data_Dslr)
data_Caltech=pca_D.fit_transform(data_Caltech)

'''预处理'''
data_Webcam=np.mat(preprocessing.scale(data_Webcam)).T   #大小均为d'*n
data_Amazon=np.mat(preprocessing.scale(data_Amazon)).T
data_Dslr=np.mat(preprocessing.scale(data_Dslr)).T
data_Caltech=np.mat(preprocessing.scale(data_Caltech)).T
scipy.io.savemat('CADW/Amazon.mat',
                 {'X': data_Amazon, 'U': U_Amazon,'Y': label_A})
scipy.io.savemat('CADW/Dslr.mat',
                 {'X': data_Dslr, 'U': U_Dslr,'Y': label_D})
scipy.io.savemat('CADW/Webcam.mat',
                 {'X': data_Webcam, 'U': U_Webcam,'Y': label_W})
scipy.io.savemat('CADW/Caltech.mat',
                 {'X': data_Caltech, 'U': U_Caltech,'Y': label_C})