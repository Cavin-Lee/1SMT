# -*- encoding:utf-8 -*-

'''
一个源域：
Caltech：类别分别为1,2,3,4,5,6,7,8,9,10
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

classnum_Caltech=10; classnum_Dslr=6; classnum_Webcam=6
M=2   #目标域个数

Caltech=scipy.io.loadmat('CADW/C--DW(2)/Caltech.mat')
Dslr=scipy.io.loadmat('CADW/C--DW(2)/Dslr(2).mat')      #第一个目标域
Webcam=scipy.io.loadmat('CADW/C--DW(2)/Webcam(2).mat')      #第二个目标域

data_Caltech=Caltech['Xs']; data_Dslr=Dslr['Xt']; data_Webcam=Webcam['Xt']
label_C=Caltech['Ys']; label_D=Dslr['Yt']; label_W=Webcam['Yt']
U_Caltech=np.mat(Caltech['Us'])

'''降维'''
pca_C=KernelPCA(n_components=80,kernel='rbf')
pca_D=KernelPCA(n_components=80,kernel='rbf')
pca_W=KernelPCA(n_components=80,kernel='rbf')

data_Webcam=pca_D.fit_transform(data_Webcam)  #大小均为n*d'
data_Caltech=pca_C.fit_transform(data_Caltech)
data_Dslr=pca_D.fit_transform(data_Dslr)

'''预处理'''
data_Webcam=np.mat(preprocessing.scale(data_Webcam)).T   #大小均为d'*n
data_Caltech=np.mat(preprocessing.scale(data_Caltech)).T
data_Dslr=np.mat(preprocessing.scale(data_Dslr)).T
scipy.io.savemat('CADW/data_C.mat',
                 {'data_Caltech': data_Caltech, 'U_Caltech': U_Caltech,'label_C': label_C})
