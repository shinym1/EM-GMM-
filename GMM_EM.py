import pywt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import math
import GMM_EM

data = pd.read_csv('hight.csv')
data = np.array(data)
data_swap=np.swapaxes(data,1,0)
data_mix = data_swap[0]
data_m = []
data_f = []
for sample in data:
    if sample[1] == 'M':
        data_m.append(sample[0])
        continue
    else:
        data_f.append(sample[0])
        continue
data_m = np.array(data_m)  #男生有78个
data_f = np.array(data_f)  #女生有17个

def get_sigma(data):
    Mu = sum(data) / data.size
    Com = np.tile(Mu,data.size)
    Sigma = np.sqrt(sum((data-Com)**2)/data.size)
    return [Mu,Sigma]




#print(get_sigma(data_mix))

def gauss(x, Mu, Sigma):
    g = []
    for i in range(Mu.size):
        g.append(np.e**(-((x-Mu[i])**2)/(2*(Sigma[i]**2)))/(np.sqrt(2*math.pi)*Sigma[i]))
    return np.array(g)


def get_gamma(X, Alpha, Mu, Sigma, K):
    m = X.size
    Gamma = np.zeros((m,K))
    for j in range(m):
        for i in range(K):
            a = Alpha[i]*gauss(X[j],Mu,Sigma)[i]
            b = np.dot(Alpha,gauss(X[j],Mu,Sigma))
            Gamma[j][i] = a/b
    return Gamma

def iter_Mu(X, Gamma, Mu):
    Gamma_swa = np.swapaxes(Gamma, 1, 0)
    for i in range(Mu.size):
        Mu[i] = np.dot(X,Gamma_swa[i])/sum(Gamma_swa[i])
    return Mu

def iter_Sigma(X, Gamma, Mu, Sigma):
    Gamma_swa = np.swapaxes(Gamma, 1, 0)
    for i in range(Mu.size):
        Sigma[i] = np.sqrt(np.dot((X-Mu[i])**2,Gamma_swa[i])/sum(Gamma_swa[i]))
    return Sigma

def iter_Alpha(Gamma):
    Gamma_swa = np.swapaxes(Gamma, 1, 0)
    for i in range(Mu.size):
        Alpha[i] = sum(Gamma_swa[i])/(Gamma_swa[i].size)
    return Alpha

def get_MLE(X,Gamma,Mu,Sigma,Alpha):
    sum = 0
    for j in range(X.size):
        sum += math.log(np.dot(Alpha, gauss(X[j], Mu, Sigma)))
    return sum


#g = gauss(175,np.array([175,172.8]),np.array([1,2.2]))
m_mu,m_sigma = get_sigma(data_m)
f_mu,f_sigma = get_sigma(data_f)



Mu = np.array([190,180],dtype=float)
Sigma = np.array([6,6],dtype=float)
K= 2
X = data_mix
Alpha = np.array([0.8,0.2])
Gamma = get_gamma(X, Alpha, Mu, Sigma, K)
MLE = get_MLE(X,Gamma,Mu,Sigma,Alpha)
M = np.zeros((50,2))
E = np.zeros((50))
S = np.zeros((50,2))


for i in range(50):
    M[i] = Mu
    E[i] = MLE
    S[i] = Sigma
    print('迭代次数：', i, '均值为：', Mu, '标准差为：', Sigma, '混合系数为：', Alpha, '似然值：', MLE, '\n')
    Gamma = get_gamma(X, Alpha, Mu, Sigma, K)
    Mu = iter_Mu(X, Gamma, Mu)
    Sigma = iter_Sigma(X, Gamma, Mu, Sigma)
    Alpha = iter_Alpha(Gamma)
    MLE = get_MLE(X,Gamma,Mu,Sigma,Alpha)


# M = np.swapaxes(M,1,0)
# S = np.swapaxes(S,1,0)
# x=range(50)
# plt.figure()
# plt.plot(M[0],label='male')
# plt.plot(M[1],label='female')
# plt.legend(loc='right')
# plt.title('Mean')
# plt.xlabel('times of iterations')
# plt.ylabel('height/cm')
# plt.savefig('Mean2.png')
#
# plt.figure()
# plt.plot(S[0],label='male')
# plt.plot(S[1],label='female')
# plt.legend(loc='right')
# plt.title('Sigmod')
# plt.xlabel('times of iterations')
# plt.savefig('Sigmod2.png')
#
# plt.figure()
# plt.plot(E)
# plt.title('MLE')
# plt.xlabel('times of iterations')
# plt.savefig('MLE2.png')
# # y=torch.arange(2002)
# # plt.plot(y, L)
# # plt.plot(y, H)
# plt.show()




