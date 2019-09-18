# -*- encoding：utf-8 -*-

import numpy as np
from scipy.linalg import _solvers
import autograd.numpy as np

from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
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
def target_WTj(data,U,WS,VT_j,V_j,D,Lambda,gama_j,beta_j):
   '''
   :param data: 样本矩阵，d*n
   :param U: 隶属度矩阵，c*n
   :param WS: 源域的权重矩阵，d*c
   :param VT_j: 源域到第j个目标域的系数矩阵，cj*c(其中cj为第j个目标域的类别数，c为源域的类别数)
   :param V_j: 目标域间公共字典的第j个系数矩阵，r*cj
   :param D: 公共字典，d*r（r为字典大小）
   :param Lambda: 权衡参数
   :param gama_j: 权衡参数
   :param beta_j: 权衡参数
   :return WT_j: 第j个目标域的权重矩阵
   '''
   d,n=data.shape
   cj=V_j.shape[1]
   I=np.mat(np.eye(d))
   temp1 = np.mat(np.zeros((d, d)))
   temp2 = np.mat(np.zeros((d, cj)))
   for k in range(cj):
      Uk = np.mat(np.diag((np.multiply(U[k, :], U[k, :])).tolist()[0]))  # 对角线元素为U的第k行元素的平方
      Lk = np.mat(np.zeros((cj, n)))
      Lk[k, :] = np.mat(np.ones((1, n)))  # 第k行元素为1，其余元素均为0
      temp1 = temp1 + data * Uk * data.T
      temp2 = temp2 + data * Uk * Lk.T
   A=(1+gama_j)*I+Lambda*temp1
   B=beta_j*VT_j*VT_j.T
   Q=Lambda*temp2+beta_j*WS*VT_j.T+gama_j*D*V_j
   WT_j=np.mat(_solvers.solve_sylvester(A,B,Q))
   return WT_j

'''计算目标域的公共字典'''
def Target_D(WT,V,gama,r):
   '''
   :param WT: 目标域的权重矩阵，共有M个，以矩阵的形式存在列表中
   :param V: 目标域公共字典的额系数矩阵，共有M个，以矩阵的形式存在列表中
   :param gama: tradeoff parameter,list
   :param r: 字典个数
   :return D: 公共字典
   
   '''

   '''
   d=WT[0].shape[0]
   M=len(WT)   #目标域的个数
   temp1=np.mat(np.zeros((d,r)))
   temp2=np.mat(np.zeros((r,r)))
   for j in range(M):
      temp1=temp1+gama[j]*WT[j]*V[j].T
      temp2=temp2+gama[j]*V[j]*V[j].T
   D=temp1*np.linalg.inv(temp2)
   return D
   '''
   d = WT[0].shape[0]
   # (1) Instantiate a manifold
   manifold = Stiefel(d, r)
   M = len(WT)  # 目标域的个数
   # (2) Define the cost function (here using autograd.numpy)
   def cost(X):
      J3 = 0;
      for j in range(M):
         J3 = np.add(J3 ,0.5 * gama[j] * np.linalg.norm(WT[j] - X @ V[j], ord=2) ** 2)
      return  J3

   def egrad(X):
      temp1 = np.mat(np.zeros((d, r)))
      temp2 = np.mat(np.zeros((r, r)))
      for j in range(M):
         temp1 = temp1 + gama[j] * WT[j] * V[j].T
         temp2 = temp2 + gama[j] * V[j] * V[j].T
      print(V[j].shape)
      G3 = temp1-X*temp2;
      return G3
   def ehess(X):

      temp2 = np.mat(np.zeros((r, r)))
      for j in range(M):

         temp2 = temp2 + gama[j] * V[j] * V[j].T
      print(V[j].shape)
      H3 = np.kron(temp2.T,-1*np.eye(d));
      return H3



   problem = Problem(manifold=manifold,cost=cost, egrad=egrad, ehess=ehess)

   # (3) Instantiate a Pymanopt solver
   solver = SteepestDescent()

   # let Pymanopt do the rest
   D = solver.solve(problem)
   return D

'''计算源域到目标域的系数矩阵VT_j'''
def Source_Target_VTj(WT_j,WS,beta,eta):
   '''
   :param WT_j:  第j个目标域权重矩阵，d*cj
   :param WS: 源域权重矩阵，d*c
   :param beta: 权衡参数
   :param eta: 权衡参数
   :return VT_j: 源域到目标域的权重系数
   '''
   '''初始化'''
   cj=WT_j.shape[1]
   J_new = float("inf")
   MT_j = np.mat(np.eye(cj))  # 初始化为单位矩阵
   VT_j = float("inf")
   while (1):
      # old_VT = VT
      VT_j = np.linalg.inv(beta * WT_j.T * WT_j + 2 * eta * MT_j) * beta * WT_j.T * WS
      '''更新矩阵MT_j'''
      for i in range(cj):
         MT_j[i, i] = (2 * np.linalg.norm(VT_j[i] + 0.000001, 2)) ** (-1)
      J = J_new
      V_sum = 0
      for i in range(cj):
         V_sum = V_sum + np.linalg.norm(VT_j[i, :], ord=2)
      J_new = 0.5 * (beta * np.linalg.norm(WS - WT_j * VT_j, ord=2) ** 2 + 2 * eta * V_sum)
      # print('VT差值',abs(J_new - J))
      if (abs((J_new - J)/J_new) < 0.001):
         break
         # if (np.sum(sum(abs(VT - old_VT)),axis=1)[0,0] < 10**(-10)):
         # break
   return VT_j

'''计算目标域间的系数矩阵V_j'''
def Target_Vj(WT_j,D,gama_j,eta_j):
   '''
   :param WT_j: 第j个目标域的权重矩阵，d*cj
   :param D: 目标域的公共参数字典，d*r
   :param gama_j: 权衡参数
   :param eta_j: 权衡参数
   :param r: 字典个数
   :return V_j: 第j个目标域的系数矩阵
   '''
   '''初始化'''
   r= D.shape[1]
   J_new = float("inf")
   M_j = np.mat(np.eye(r))  # 初始化为单位矩阵
   V_j = float("inf")
   while (1):
      # old_VT = VT
      V_j = np.linalg.inv(gama_j * D.T @ D + 2 * eta_j * M_j) * gama_j * D.T @ WT_j
      '''更新矩阵MT_j'''
      for i in range(r):
         M_j[i, i] = (2 * np.linalg.norm(V_j[i] + 0.000001, 2)) ** (-1)
      J = J_new
      V_sum = 0
      for i in range(r):
         V_sum = V_sum + np.linalg.norm(V_j[i, :], ord=2)
      J_new = 0.5 * (gama_j * np.linalg.norm(WT_j - D * V_j, ord=2) ** 2 + 2 * eta_j * V_sum)
      # print('VT差值',abs(J_new - J))
      if (abs((J_new - J)/J_new) < 0.001):
         break
         # if (np.sum(sum(abs(VT - old_VT)),axis=1)[0,0] < 10**(-10)):
         # break
   return V_j

