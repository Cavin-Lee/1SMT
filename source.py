# -*-encoding:utf-8 -*-

import numpy as np

'''对于隶属度矩阵'''
def Normization(U):
    '''
    :param U:样本的隶属度矩阵
    :return nor_U:归一化的隶属度矩阵
    '''
    U_colum=sum(U)
    nor_U=U/U_colum
    return nor_U

'''计算隶属度矩阵，设定模糊系数m=2'''
def SLMC_U(data,classnum,W):
   '''
   :param data: 输入数据，d*n
   :param classnum: 类的个数，即簇数
   :param W: 权重矩阵，d*c
   :return: U_new（隶属度矩阵，c*n）
   '''
   '''初始化'''
   d,n=data.shape     #d为维度，n为样本个数
   Y=np.mat(np.eye(classnum,dtype=int))    #生成classnum*classnum的矩阵
   U=np.zeros((classnum,n))    #定义U矩阵，c*n

   '''获得矩阵U'''
   fdata=W.T*data     #计算f(X)=W^T*X
   for k in range(classnum):
      for i in range(n):
         U[k,i]=np.linalg.norm(fdata[:,i]-Y[:,k],ord=2)**(-2)
   temp=np.mat(np.ones(classnum)).T*np.sum(U,axis=0)     #矩阵按列求和
   U_new=U/temp
   return U_new

'''计算权重矩阵W，设定模糊系数m=2'''
def SLMC_W(data,U,Lambda,classnum):
   '''
   :param data: 输入数据，d*n
   :param U: 隶属度矩阵，c*n
   :param Lambda: 超参数
   :param classnum: 类的个数，即簇数
   :return: WS（源域的权重矩阵，d*c）
   '''
   '''初始化'''
   d,n=data.shape        #d为维度，n为样本个数
   temp1=np.mat(np.zeros((d,d)))      #闭式解前半部分
   temp2=np.mat(np.zeros((d,classnum)))        #闭式解后半部分
   #I = np.ones((d, d))  #生成d*d的单位矩阵
   for k in range(classnum):
      Uk=np.mat(np.diag((np.multiply(U[k,:],U[k,:])).tolist()[0]))       #对角线元素为U的第k行元素的平方
      Lk=np.mat(np.zeros((classnum,n)))
      Lk[k,:]=np.mat(np.ones((1,n)))     #第k行元素为1，其余元素均为0
      temp1=temp1+data*Uk*data.T
      temp2=temp2+data*Uk*Lk.T
   WS=np.linalg.inv(1+Lambda*temp1)*Lambda*temp2
   '''
   S=np.zeros([temp1.shape[0],temp1.shape[0]])
   TEMP=1+Lambda*temp1
   u, sigma, vt=np.linalg.svd(TEMP)
   for i in range(TEMP.shape[0]):
       S[i][i]=sigma[i]**(-1)
   inv_temp=u*S.T*vt
   WS=inv_temp*(Lambda*temp2)
   '''
   return WS
