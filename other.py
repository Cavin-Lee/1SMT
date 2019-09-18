

'''
YS_Amazon=(W_Amazon.T*data_Amazon).T
raw_A,column_A=YS_Amazon.shape
Amazon_pre=[]

for i in range(raw_A):
    _positon=np.argmax(YS_Amazon[i])
    m,n=divmod(_positon,column_A)
    Amazon_pre.append(n+1)
NMI_A=metrics.normalized_mutual_info_score(label_Amazon,Amazon_pre)
RI_A=RandIndex.rand_index_score(label_Amazon,Amazon_pre)
ACC_A=metrics.accuracy_score(label_Amazon,Amazon_pre)
print('NMI大小：',NMI_A)
print('RI大小：',RI_A)
print('ACC大小：',ACC_A)

YS_Caltech=(W_Caltech.T*data_Caltech).T
raw_C,column_C=YS_Caltech.shape
Caltech_pre=[]

for i in range(raw_C):
    _positon=np.argmax(YS_Caltech[i])
    m,n=divmod(_positon,column_C)
    Caltech_pre.append(n+1)
print(label_Caltech)
print(Caltech_pre)
NMI_C=metrics.normalized_mutual_info_score(label_Caltech,Caltech_pre)
RI_C=RandIndex.rand_index_score(label_Caltech,Caltech_pre)
ACC_C=metrics.accuracy_score(label_Caltech,Caltech_pre)
print('NMI大小：',NMI_C)
print('RI大小：',RI_C)
print('ACC大小：',ACC_C)
'''