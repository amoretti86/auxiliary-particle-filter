'''
Implements the Fully Adapted Auxiliary Particle Filter
as described by Whiteley and Johansen 

Algorithm 2 on page 5 of Chapter 3 in Recent Developments in 
Auxiliary Particle Filtering: http://www.maths.bris.ac.uk/~manpw/apf_chapter.pdf

Demo is included below for a two dimensional state space model with
Gaussian process and measurements. 

By Antonio Moretti: amoretti@cs.columbia.edu
'''

import math
import numpy as np
import statespace as ss
import scipy as sp
from scipy.stats import norm
from scipy import misc

def make_mvn_pdf(mu, sigma):
'''
Defines a multivariate Gaussian density
'''
    def f(x):
        return sp.stats.multivariate_normal.pdf(x, mu, sigma)
    return f

def make_poisson(k):
'''
Defines a multivariate (uncorrelated) Poisson density
'''
    def f(theta):
        prob = 1
        for i in range(len(k)):
            prob *= sp.stats.poisson.pmf(k[i], np.exp(theta[i]))
        return prob
    return f

def fhn(Y, deltat):
'''
Discretize Fitzhugh Nagumo Dynamics via First Order Euler method
'''
    y1 = Y[0]
    y2 = Y[1]
    return [y1 + (y1 - (y1**3)/3 - y2)*deltat, y2 - (0.08*(y1 + 0.7 - 0.8*y2))*deltat]

def integrate_gaussian(grid, weight, mean, std, f):
    '''
    Performs gaussian quadrature via hermite polynomials
    '''
    sq2=np.sqrt(2)
    zz = sq2*std*grid + mean
    value = np.dot(f(zz), weight)/np.sqrt(np.pi)
    return value

def bivariate_gauss_hermite(xt, wt, mean, T, gfunc, XX, YY):
    '''
    Performs two dimensional Gauss-Hermite Quadrature with a change of measure to account for mu and sigma
    '''
    import scipy as sp
    from scipy import linalg
    mat = np.array([XX.flatten(), YY.flatten()]).T
    grid_trans = np.sqrt(2)*np.dot(mat , T) + mean
    geval = np.asarray([gfunc(xx) for xx in grid_trans]).reshape([len(xt), len(xt)])
    c = 1/(np.pi)

    y_marginal = np.zeros(len(xt))
    for idx in xrange(len(xt)):
        y_marginal[idx] = np.dot(geval[idx,:], wt)

    theta = np.dot(y_marginal, wt)*c
    return theta

def apf(obs, time, n_particles, n_gridpoints, A, B, Sigma, Gamma, x_0):
    '''
    Implements the Auxiliary Particle Filter as described by Whiteley and Johansen
    Algorithm 2 on page 5: http://www.maths.bris.ac.uk/~manpw/apf_chapter.pdf

    Input: a time series of observed signals, a time length
    number of particles, parameters for the process model and
    measurement model (means and variances), and initial value

    Output: weights (W) and trajectories (X)
    '''
    assert(len(obs) == time)

    # Initialize variables
    dimension = B.shape[1]
    n_gridpoints = n_gridpoints

    X = np.zeros((n_particles, time, dimension))
    W = np.zeros((n_particles, time))
    k = np.zeros((n_particles, time))
    proposal_covariance_matrix = np.eye(dimension)

    [xt, wt] = np.polynomial.hermite.hermgauss(n_gridpoints)
    # need to generalize the computation of posterior integral
    XX = np.tile(xt.reshape([1, len(xt)]), [len(xt), 1])
    YY = np.tile(xt.reshape([len(xt), 1]), [1, len(xt)])
    T = sp.linalg.sqrtm(Sigma)

    for i in range(0,n_particles):
        X[i,0,:] = np.random.randn(1,dimension)[0] 
        g = make_mvn_pdf(x_0, Gamma)(np.dot(B,X[i,0,:]))
        nu = make_mvn_pdf(x_0, Sigma)(X[i,0,:])
        q = make_mvn_pdf(np.zeros(dimension), proposal_covariance_matrix)(X[i,0,:])
        W[i,0] = g*nu/q

    # Normalize weights
    
    #pdb.set_trace()
    for t in range(1, time):

        # Update weights based on integral
        for i in range(N_particles):
            # Compute the posterior integral p(y_n | x_{n-1})
            g_int_func = make_mvn_pdf(np.dot(B,obs[t,:]),Gamma)
            k[i,t] = bivariate_gauss_hermite(xt, wt, np.dot(A,X[i,t-1,:]), T, g_int_func, XX, YY)
            #print "posterior integral: ", k[i,t]
            W[i,t-1] = W[i,t-1]*k[i,t]

        # Resample
        Xprime = np.random.choice(N_particles, N_particles, p = W[:,t-1]/np.sum(W[:,t-1]), replace = True)
        Xtilde = [X[i,t-1,:] for i in Xprime]
        # Reset weights and particles

        for i in range(N_particles):
            # Select new particles            
            X[i,t-1,:] = Xtilde[i]
            # Resample particles and reset weights
            X[i,t,:] = np.random.randn(1,dimension)[0] + X[i,t-1,:]
            #print "particles: ", X[i,t,:]
            g = make_mvn_pdf(np.dot(B,X[i,t,:]),Gamma)(obs[t,:])
            q = make_mvn_pdf(X[i,t-1,:],np.identity(dimension))(X[i,t,:])
            f = make_mvn_pdf(np.dot(A,X[i,t-1,:]), Sigma)(X[i,t,:])
            W[i,t] = (g*f)/(k[i,t]*q)

        print "time: ", t
    return W, X, k

if __name__ == "__main__":

    import mvgauss as mvg
    import numpy as np

    A = np.diag([.8, .3])
    Sigma = .5*np.asarray([[1, .5],[.5, 1]])
    Gamma = np.asarray([[1, .45], [.45, 1]])
    B = np.diag([2, 2])
    T = 50
    x_0 = np.random.randn(1, 2)[0]


    import matplotlib.pyplot as plt
    [hidden, obs] = mvg.simulateStateSpace(A,B,Sigma,Gamma,T,x_0)
    n_particles = 500
    [w, x, k] = apf(obs, T, n_particles, 10, A, B, Sigma, Gamma, x_0)

    # visualize parameters
    
    plt.subplot(141)
    plt.imshow(w)
    plt.xlabel('time')
    plt.ylabel('particle weights')
    plt.title('weight matrix')
    plt.subplot(142)
    plt.imshow(x[:,:,0])
    plt.xlabel('time')
    plt.ylabel('particles')
    plt.title('path matrix')
    plt.subplot(143)
    plt.imshow(x[:,:,1])
    plt.xlabel('time')
    plt.ylabel('particles')
    plt.title('path matrix')
    plt.subplot(144)
    plt.imshow(k)
    plt.xlabel('time')
    plt.ylabel('p(y_n | x_{n-1})')
    plt.title('posterior')
    plt.subplots_adjust(right=2.5, hspace=.75)

    # examine particle trajectories over time
    plt.subplot(141)
    plt.plot(np.transpose(x[:,:,0]), alpha=.01, linewidth=1.5)
    plt.xlabel('time')
    plt.ylabel('displacement')
    plt.title('particle path trajectories over time (dim 1)')
    
    plt.subplot(142)
    plt.plot(np.transpose(x[:,:,1]), alpha=.01, linewidth=1.5)
    plt.xlabel('time')
    plt.ylabel('displacement')
    plt.title('particle path trajectories over time (dim 2)')
    

    plt.subplot(143)
    plt.plot(x[:,:,0])
    plt.xlabel('particle')
    plt.ylabel('time')
    plt.title('particle variance (dim 1)')
    plt.subplot(144)

    plt.plot(x[:,:,1])
    plt.xlabel('particle')
    plt.ylabel('time')
    plt.title('particle variance (dim 2)')
    plt.subplots_adjust(right=2.5, hspace=.85)

    # average over particle trajectories to obtain predicted state means for APF output
    predsignal1 = np.mean(x[:,:,0], axis=0)
    predsignal2 = np.mean(x[:,:,1], axis=0)

    # check predicted vs true signal
    time = np.arange(T)
    plt.subplot(121)
    plt.title('apf')
    lo = plt.plot(time, hidden[:,0], 'r', time, predsignal1, 'b')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.legend(lo, ('true value','prediction'))

    plt.subplot(122)
    plt.title('apf')
    lo = plt.plot(time, hidden[:,1], 'r', time, predsignal2, 'b')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.legend(lo, ('true value','prediction'))
    plt.subplots_adjust(right=1.5, hspace=.75)


    from pykalman import KalmanFilter
    # run kalman filter and check parameters
    kf = KalmanFilter(transition_matrices = A, observation_matrices = B)
    kf = kf.em(obs, n_iter=50)
    (filtered_state_means, filtered_state_covariances) = kf.filter(obs)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(obs)
    # check true values against standard kalman filter output
    plt.subplot(221)
    lo = plt.plot(time, hidden[:,0], 'r', time, filtered_state_means[:,0], 'b')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.title('kalman filter (dim 1)')
    plt.legend(lo, ('true value','prediction'), loc='lower left')

    plt.subplot(223)
    plt.title('apf (dim 1)')
    lo = plt.plot(time, hidden[:,0], 'r', time, predsignal1, 'b')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.legend(lo, ('true value','prediction'), loc='lower left')
    plt.subplots_adjust(right=1.5, hspace=.75)

    plt.subplot(222)
    lo = plt.plot(time, hidden[:,1], 'r', time, filtered_state_means[:,1], 'b')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.title('kalman filter (dim 2)')
    plt.legend(lo, ('true value','prediction'), loc='lower left')

    plt.subplot(224)
    plt.title('apf (dim 2)')
    lo = plt.plot(time, hidden[:,1], 'r', time, predsignal2, 'b')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.legend(lo, ('true value','prediction'), loc='lower left')
    plt.subplots_adjust(right=1.75, hspace=.85)
