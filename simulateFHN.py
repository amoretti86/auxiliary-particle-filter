'''
Simulates data for the FitsHugh Nagumo (FHN) Biophysical model by solving the resulting differential equations
Parameter values are hard coded according to Scholarpedia (see below) but can easily be changed. Note ode45 is
nondeterministic and sometimes returns null values. 

http://www.scholarpedia.org/article/FitzHugh-Nagumo_model

By Antonio Moretti amoretti@cs.columbia.edu

dV/dt = V(a-V)(V-1) - w + I
dw/dt = bV - cw
'''

import scipy as sp
import pylab as plt
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
import matplotlib.gridspec as gridspec


def simFN(a, b, t, disp, I):
    '''
    Integrates the FHN ODEs

    Input:
    a: the shape of the cubic parabola
    b: describes the kinetics of the recovery variable w
    c: describes the kinetics of the recovery variable
    t: time to integrate over
    disp: (True/False) plot data
    I: input current

    Output
    V - membrane voltage
    w - recovery variable that mimics activation of an outward current
    I - resting current
    '''
    def dALLdt(X, t):

        V, w, a, b, c, I = X

        dVdt = V-V**3/3 - w + I
        dwdt = 0.08*(V + 0.7 - 0.8*w)
        return [dVdt, dwdt]

    X = odeint(dALLdt, [0, 0.05, a, b, 0.5, I], t)
    V = X[:,0]
    w = X[:,1]

    if disp == True:

        plt.subplot(211)
        plt.title('FitzHugh-Nagumo')
        plt.plot(t, V, 'r', label = 'v')
        plt.ylabel('V (mV)')
        plt.axis('tight')
        plt.subplot(212)
        plt.plot(t, w, 'g', label = 'w')
        plt.ylabel('w')
        plt.show()

        def f(Y, t):
            # Euler discretization
            y1, y2 = Y
            return [y1 - (y1**3)/3 - y2, 0.08*(y1 + 0.7 - 0.8*y2)]
        
        # plot phase portraint of the ODE system
        y1 = np.linspace(-5.0, 5.0, 20)
        y2 = np.linspace(-20.0, 20.0, 20)
        Y1, Y2 = np.meshgrid(y1, y2)
        tau = 0
        u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
        NI, NJ = Y1.shape
        
        for i in range(NI):
            for j in range(NJ):
                x = Y1[i,j]
                y = Y2[i,j]
                yprime = f([x,y], tau)
                u[i,j] = yprime[0]
                v[i,j] = yprime[1]

        Q = plt.quiver(Y1, Y2, u, v, color = 'r')
        plt.xlabel('V')
        plt.ylabel('w')
        plt.title('Phase Portrait of Fitzhugh Nagumo System')
        plt.show()

    return V, w

if __name__ == "__main__":
    
    t = sp.arange(0.0, 100.0, 0.01)
    a = 0.7
    b = 0.8
    [V,w] = simFN(a,b,t,True, 1)

