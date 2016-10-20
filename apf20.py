'''
Implements the Fully Adapted Auxiliary Particle Filter as described by Whiteley and Johansen
Chapter 3 Algorithm 2 on page 5 of Recent Developments in Auxiliary Particle Filtering:
http://www.maths.bris.ac.uk/~manpw/apf_chapter.pdf

The algorithm is modified to perform inference in nonlinear dynamical systems and supports
the Fitzhugh Nagumo model (or Van der Pol Oscillator)

by Antonio Moretti - amoretti@cs.columbia.edu

'''

import math
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy import misc

def make_mvn_pdf(mu, sigma):
    def f(x):
        return sp.stats.multivariate_normal.pdf(x, mu, sigma)
    return f

def make_poisson(k):
    def f(theta):
        prob = 1
        for i in range(len(k)):
            prob *= sp.stats.poisson.pmf(k[i], np.exp(theta[i]))
        return prob
    return f

def fhn(Y, deltat, I):
    y1 = Y[0]
    y2 = Y[1]
    return [y1 + (y1 - (y1**3)/3 - y2 + I)*deltat, y2 + (0.08*(y1 + 0.7 - 0.8*y2))*deltat]

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

def apf(obs, time, n_particles, n_gridpoints, B, Sigma, Gamma, x_0, I_ext):
    '''
    Implements the Auxiliary Particle Filter as described by Whiteley and Johansen
    Algorithm 2 on page 5: http://www.maths.bris.ac.uk/~manpw/apf_chapter.pdf

    Input: 
    (obs):          a time x dimension matrix representing a time series of observed signals
    (time):         a scalar representing the corresponding time length
    (n_particles):  a scalar representing the number of particles to use in the simulation
    (n_gridpoints): a scalar representing the number of grid points or nodes to use for the quadrature
    (B):            a 2x2 propagator matrix for the dynamics
    (Sigma):        a 2x2 covariance matrix
    (Gamma):        a 2x2 covariance matrix
    (x_0):          a 2x1 vector of representing the initial value of the signal
    (I_ext):        a scalar representing input current magnitude

    Output: 
    (W) : an n_particles x time matrix of weights
    (X) : an n_particles x time x dimension tensor of trajectories
    (k) : an n_particles x time matrix of the posterior integral for each particle at each time point

    Averaging the trajectory tensor (X) across particles approximates the functional integral. Smooth the
    resulting signal to remove noise.
    '''
    assert(len(obs) == time)

    # Initialize variables
    dimension = 2
    n_gridpoints = n_gridpoints

    X = np.zeros((n_particles, time, dimension))
    W = np.zeros((n_particles, time))
    k = np.zeros((n_particles, time))
    proposal_covariance_matrix = 0.075*np.eye(dimension)
    delta_t = 0.25

    [xt, wt] = np.polynomial.hermite.hermgauss(n_gridpoints)
    # TO DO: generalize the computation of posterior integral
    XX = np.tile(xt.reshape([1, len(xt)]), [len(xt), 1])
    YY = np.tile(xt.reshape([len(xt), 1]), [1, len(xt)])
    T = sp.linalg.sqrtm(Sigma)

    # sample particles and weights at time 1
    import pdb
    for i in range(0,n_particles):
        X[i,0,:] = np.random.randn(1,dimension)[0] 
        g = make_mvn_pdf(x_0, Gamma)(obs[0,:])#(np.dot(B,X[i,0,:])).ravel()
        nu = make_mvn_pdf(x_0, Sigma)(X[i,0,:])
        q = make_mvn_pdf(np.zeros(dimension), proposal_covariance_matrix)(X[i,0,:])
        W[i,0] = g*nu/q
    
    # main loop of program at time > 1
    for t in range(1, time):

        # Update weights and propagate particles based on postrior integral
        for i in range(n_particles):
            # Compute the posterior integral p(y_n | x_{n-1})
            g_mean = np.dot(B,X[i,t,:])
            g_int_func = make_mvn_pdf(g_mean,Gamma)
            k[i,t] = bivariate_gauss_hermite(xt, wt, fhn(X[i,t-1,:],delta_t,I_ext), T, g_int_func, XX, YY)
            # Reweight particles
            W[i,t-1] = W[i,t-1]*k[i,t]

        # Resample
        Xprime = np.random.choice(n_particles, n_particles, p = W[:,t-1]/np.sum(W[:,t-1]), replace = True)
        Xtilde = [X[i,t-1,:] for i in Xprime]

        # Reset weights and particles
        for i in range(n_particles):
            # Select new particles            
            X[i,t-1,:] = Xtilde[i]
            # Resample particles and reset weights
            X[i,t,:] = np.random.randn(1,dimension)[0] + X[i,t-1,:]
            # Update proposal and target distributions
            reshaped_g_mean = np.dot(B,X[i,t,:]).ravel()
            g = make_mvn_pdf(reshaped_g_mean,Gamma)(obs[t,:])
            q = make_mvn_pdf(X[i,t-1,:],proposal_covariance_matrix)(X[i,t,:])
            f = make_mvn_pdf(fhn(X[i,t-1,:],delta_t,I_ext),Sigma)(X[i,t,:])
            # Update weights
            W[i,t] = (g*f)/(k[i,t]*q)

        print "time: ", t
    return W, X, k


