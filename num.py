
import scipy.io
'''
digits=scipy.io.loadmat('DMU/digits.mat')
y1=digits['Ys'].T
y1=list(y1[0])
for i in range(1,11):
    print(y1.count(i))
print()
print()
mnist=scipy.io.loadmat('DMU/mnist.mat')

y1=mnist['Ys'].T
y1=list(y1[0])
for i in range(1,11):
    print(y1.count(i))

print()
print()
usps=scipy.io.loadmat('DMU/usps.mat')
y1=usps['Ys'].T
y1=list(y1[0])
for i in range(1,11):
    print(y1.count(i))
'''

Y5=scipy.io.loadmat('YaleB/YaleB_32x32.mat')
y1=Y5['gnd'].T
y1=list(y1[0])
for i in range(1,39):
    print(y1.count(i))
