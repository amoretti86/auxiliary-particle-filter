'''
simulte data to test the auxiliary particle filter according to the following recursions:

x_1 = x_0 + \sigma \epsilon_0
x_t = a*x_{t-1} + \sigma*\epsilon_t
y_t = b*x_t + \gamma*\eta_t

\epsilon_t ~ N(0, 1)
\eta_t ~ N(0, 1)

parameters:
a, b, \sigma, \gamma, T, x_0

'''

import numpy as np
import random
import matplotlib.pyplot as plt

def simulateStateSpace(A,B,Sigma,T,x_0, plot = True):
    '''
    Generates synthetic data for a state space model according to the recursions above.
    a and sigma are the terms acting on x and epsilon
    b and gamma are the terms acting on y and eta
    epsilon and gamma are standard white noise
    y is a hidden variable linearly dependent upon x the noisy observations
    '''

    if Sigma.any < 0:
        print "Sigma must be PSD"
        return

    X = np.zeros((T,2))
    Y = np.zeros((T,2))
    time = np.arange(T)
    X[0] = x_0#np.random.normal(0,1,1)#x_0
    CS = np.linalg.cholesky(Sigma)

    for i in range(1,T):
        #X[i] = a*X[i-1] + sigma*np.random.normal(0,1,1)
        #X[i,:] = np.dot(A, X[i-1,:])*np.random.randn(1,2)[0]
        #Y[i] = np.random.poisson(np.exp(b*X[i]))
        X[i,:] = np.dot(A, X[i-1,:])+np.dot(CS,np.random.randn(1,2)[0])
        Y[i,:] = np.random.poisson(np.exp(np.dot(B,X[i,:])))
        #Y[i,1] = np.random.poisson(np.exp(B[1]*X[i,1]))

    if plot == True:
        #plt.subplot(121)
        
        lo = plt.plot(time, X[:,0], 'r', time, Y[:,0], 'b')
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title('signal in first dim')
        plt.legend(lo, ('X', 'Y'))
        plt.show()
        #plt.subplot(122)
        lo = plt.plot(time, X[:,1], 'r', time, Y[:,1], 'b')
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title('signal in second dim')
        plt.legend(lo, ('X', 'Y'))
        #plt.subplots_adjust(right=1.5, hspace=.75)
        plt.show()

    return X, Y

if __name__ == "__main__":
    '''
    a = .9
    b = 0.25
    sigma = .5
    #gamma = .24
    T = 1000
    x_0 = np.random.normal(0,1,1)
    print x_0[0]
    [hidden, obs] = simulateStateSpace(a,b,sigma,T, x_0)#x_0)
    '''
    A = np.asarray([.65, .24])
    Sigma = np.asarray([[1, .5],[.5, 1]])
    #print A
    #print sigma
    B = np.asarray([.24, .26])
    T = 100
    x_0 = np.random.randn(1, 2)[0]
    print x_0
    [hidden, obs] = simulateStateSpace(A,B,Sigma,T,x_0)
