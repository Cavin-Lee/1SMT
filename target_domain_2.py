# -*- encoding：utf-8 -*-

import numpy as np
from scipy.linalg import _solvers

'''计算隶属度矩阵，设定模糊系数m=2'''
def target_U(data,classnum,WT):
   '''
   :param data: 输入数据，d*n
   :param classnum: 类的个数，即簇数
   :param WT: 目标域权重矩阵，d*c
   :return: U_new（隶属度矩阵，c*n）
   '''
   '''初始化'''
   d,n=data.shape     #d为维度，n为样本个数
   Y=np.mat(np.eye(classnum,dtype=int))    #生成classnum*classnum的矩阵
   U=np.zeros((classnum,n))    #定义U矩阵，c*n

   '''获得矩阵U'''
    #计算f(X)=W^T*X
   fdata=WT.T*data
   for k in range(classnum):
      for i in range(n):
         U[k,i]=(np.linalg.norm(fdata[:,i]-Y[:,k],ord=2))**(-2)
   temp=np.mat(np.ones(classnum)).T*np.sum(U,axis=0)     #矩阵按列求和
   U_new=U/temp
   return U_new

'''计算第j个目标域的权重'''
def target_WTj(data,U,WS,AT,Lambda,beta):
   '''
   :param data: 样本矩阵，d*n
   :param U: 隶属度矩阵，c*n
   :param A: 源域参数
   :param Lambda: 权衡参数
   :param gama: 权衡参数
   :param beta: 权衡参数

   '''
   d,n=data.shape
   cj,heh=AT.shape
   I=np.mat(np.eye(d))
   temp1 = np.mat(np.zeros((d, d)))
   temp2 = np.mat(np.zeros((d, cj)))
   for k in range(cj):
      Uk = np.mat(np.diag((np.multiply(U[k, :], U[k, :])).tolist()[0]))  # 对角线元素为U的第k行元素的平方
      Lk = np.mat(np.zeros((cj, n)))
      Lk[k, :] = np.mat(np.ones((1, n)))  # 第k行元素为1，其余元素均为0
      temp1 = temp1 + data * Uk * data.T
      temp2 = temp2 + data * Uk * Lk.T
   WT_j= np.linalg.inv(1 + Lambda * temp1-beta) * (Lambda * temp2+beta*WS*AT.T)

  # A=I+Lambda*temp1

  # B=beta*np.dot(AT,AT.T)
  # Q=Lambda*temp2+beta*WS*AT.T
   #WT_j=np.mat(_solvers.solve_sylvester(A,B,Q))

   return WT_j

'''计算目标域间的系数矩阵A'''
def Target_A( WS, WT,K):
    dS,cS=WS.shape
    dT,cT=WT.shape

    B=np.zeros((cS,cT))
    L=np.zeros((cS,cT))
    LL=np.zeros((1, cS))
    for k in range(cS):
        for i in range(cT):
            L[k,i]=(np.linalg.norm(WS[:,k]-WT[:,i],ord=2))
    LL=np.min(L,1);'''计算所有行的最小值'''
    min_loc=np.argmin(L,1);'''计算所有行的最小值'''
    sortLL=sorted(LL);
    tmp=np.array(LL<=sortLL[K-1],dtype='bool');
    source_loc=np.nonzero(tmp);
    source_loc=np.array(source_loc,dtype='int')
    # print(source_loc)
    for i in range(K):
          B[source_loc[:,i],min_loc[source_loc[:,i]]]=1
    A=B.T
    A=np.array(A,dtype='double')
    return A