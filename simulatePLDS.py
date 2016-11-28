'''
Creates synthetic data for the Poisson Linear Dynamical System according to the following recursions:

x_t = Ax_{t-1} + Sigma^{1/2}epsilon    where x ~ N(Ax_{t,1}, Sigma^{1/2})
y_t = Pois(exp(Bx_t)) + \eta            where y ~ Pois(exp(Bx_t))

by Antonio Moretti - amoretti@cs.columbia.edu
'''

import numpy as np
import random
import matplotlib.pyplot as plt

def simulateStateSpace(A,B,Sigma,T,x_0, plot = True):
    '''
    Generates synthetic data for a state space model according to the recursions above.
    '''
    if Sigma.any < 0:
        print "Sigma must be PSD"
        return

    X = np.zeros((T,2))
    Y = np.zeros((T,2))
    time = np.arange(T)
    X[0] = x_0
    CS = np.linalg.cholesky(Sigma)

    for i in range(1,T):
        X[i,:] = np.dot(A, X[i-1,:])+np.dot(CS,np.random.randn(1,2)[0])
        Y[i,:] = np.random.poisson(np.exp(np.dot(B,X[i,:])))

    if plot == True:
        
        plt.subplot(121)
        lo = plt.plot(time, X[:,0], 'r', time, Y[:,0], 'b')
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title('signal in first dim')
        plt.legend(lo, ('process (X)', 'measurement (Y)'))
        plt.subplot(122)
        lo = plt.plot(time, X[:,1], 'r', time, Y[:,1], 'b')
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title('signal in second dim')
        plt.legend(lo, ('process (X)', 'measurement (Y)'))
        #plt.subplots_adjust(right=1.5, hspace=.75)
        plt.show()

    return X, Y

if __name__ == "__main__":

    A = np.asarray([.65, .24])
    Sigma = np.asarray([[1, .5],[.5, 1]])
    B = np.asarray([.24, .26])
    T = 100
    x_0 = np.random.randn(1, 2)[0]
    [hidden, obs] = simulateStateSpace(A,B,Sigma,T,x_0)
