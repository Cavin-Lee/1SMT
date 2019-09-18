# -*-encoding:utf-8 -*-

import numpy as np

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

'''计算权重矩阵W，设定模糊系数m=2'''
def target_WT(data,U,Lambda,beta2,classnum,A,VT):
   '''
   :param data: 输入数据，d*n
   :param U: 隶属度矩阵，c*n
   :param Lambda: 超参数
   :param beta2: 超参数
   :param classnum: 类的个数，即簇数
   :param A: 源域和目标域中共享的基A，d*r       可以将这里的r设置为c吗？这样是否可以实现字典学习，还是说是人为自己设置
   :param VT: 系数矩阵，满足行稀疏，r*c
   :return: WT（目标域的权重矩阵，d*c）
   '''
   '''初始化'''
   d,n=data.shape        #d为维度，n为样本个数
   temp1=np.mat(np.zeros((d,d)))      #闭式解前半部分
   temp2=np.mat(np.zeros((d,classnum)))         #闭式解后半部分
   #I = np.ones((d,d))       #生成d*d的全为1的矩阵
   for k in range(classnum):
      Uk=np.mat(np.diag((np.multiply(U[k,:],U[k,:])).tolist()[0]))       #对角线元素为U的第k行元素的平方
      Lk=np.mat(np.zeros((classnum,n)))
      Lk[k,:]=np.mat(np.ones((1,n)))     #第k行元素为1，其余元素均为0
      temp1=temp1+data*Uk*data.T
      temp2=temp2+data*Uk*Lk.T
   WT=np.linalg.inv((1+beta2)+Lambda*temp1)*(Lambda*temp2+beta2*A*VT)
   return WT

'''计算源域和目标域的系数矩阵VS，VT'''
def target_VT(alpha,beta2,A,WT,r):
    '''
    :param alpha: 超参数
    :param beta2: 超参数
    :param A: 源域与目标域共享的基矩阵  d*r
    :param WT: 目标域权重矩阵    d*c
    :param r:
    :return: VS，VT    r*c
    '''
    '''初始化'''
    J_new = float("inf")
    MT=np.mat(np.eye(r))      #初始化为单位矩阵，r*r
    VT = float("inf")
    while(1):
        #old_VT = VT
        VT=np.linalg.inv(beta2*A.T*A+2*alpha*MT)*beta2*A.T*WT
        '''更新矩阵MS,MT'''
        for i in range(r):
            MT[i,i]=(2*np.linalg.norm(VT[i]+0.000001,2))**(-1)
        J = J_new
        V_sum = 0
        for i in range(r):
            V_sum = V_sum + np.linalg.norm(VT[i, :], ord=2)
        J_new = 0.5 * (beta2 * np.linalg.norm(WT - A * VT, ord=2) ** 2 + 2 * alpha * V_sum)
        #print('VT差值',abs(J_new - J))
        if (abs((J_new - J)/J_new) < 0.001):
            break
        #if (np.sum(sum(abs(VT - old_VT)),axis=1)[0,0] < 10**(-10)):
            #break
    return VT

def source_VS(alpha,beta1,A,WS,r):
    '''
    :param alpha: 超参数
    :param beta1: 超参数
    :param A: 源域与目标域共享的基矩阵  d*r
    :param WS: 源域权重矩阵    d*c
    :param r:
    :return: VS    r*c
    '''
    '''初始化'''
    J_new = float("inf")
    MS=np.mat(np.eye(r))               #初始化为单位矩阵，r*r
    VS  = float("inf")
    while(1):
        #old_VS = VS
        VS=np.linalg.inv(beta1*A.T*A+2*alpha*MS)*beta1*A.T*WS
        '''更新矩阵MS,MT'''
        for i in range(r):
            MS[i,i]=(2*np.linalg.norm(VS[i]+0.000001,2))**(-1)
        J = J_new
        V_sum = 0
        for i in range(r):
            V_sum = V_sum + np.linalg.norm(VS[i, :], ord=2)
        J_new = 0.5 * (beta1 * np.linalg.norm(WS - A * VS, ord=2) ** 2+ 2 * alpha * V_sum)
        #print('VS差值',abs(J_new - J))
        if (abs((J_new - J)/J_new) < 0.001):
            break
        #if (np.sum(sum(abs(VS - old_VS)),axis=1)[0,0] < 10**(-10)):
            #break
    return VS

'''计算源域与目标域共享的基矩阵A'''
def source_A_target(beta1,beta2,WS,WT,VS,VT):
    '''
    :param beta1: 超参数
    :param beta2: 超参数
    :param WS: 源域权重矩阵
    :param WT: 目标域权重矩阵
    :param VS: 源域系数矩阵
    :param VT: 目标域系数矩阵
    :return: A
    '''
    A=(beta1*WS*VS.T+beta2*WT*VT.T)*np.linalg.inv(beta1*VS*VS.T+beta2*VT*VT.T)
    return A