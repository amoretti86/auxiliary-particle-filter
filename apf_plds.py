'''
An implementation of the Fully Adapted Auxiliary Particle Filter as described by Whiteley and Johansen
Chapter 3 Algorithm 2 on page 5 of Recent Developments in Auxiliary Particle Filtering:
http://www.maths.bris.ac.uk/~manpw/apf_chapter.pdf

Quadrature is defined for a 2 dimensional state space model according to
Gaussian process and Poisson measurements [Poisson Linear Dynamical System (PLDS)]

by Antonio Moretti amoretti@cs.columbia.edu
'''

import math
import numpy as np
import statespace as ss
import scipy as sp
from scipy.stats import norm
from scipy import misc

def make_mvn_pdf(mu, sigma):
    
    ''' creates a multivariate gaussian pdf '''
    
    def f(x):
        return sp.stats.multivariate_normal.pdf(x, mu, sigma)
    return f

def make_poisson(k):
    
    ''' creates a multivariate poisson pmf '''
    
    def f(theta):
        prob = 1
        for i in range(len(k)):
            prob *= sp.stats.poisson.pmf(k[i], np.exp(theta[i]))
        return prob
    return f

def make_pois(k):
    
    ''' creates a univariate poisson pmf (not used) '''
    
    def f(theta):
        return sp.stats.poisson.pmf(k[0], np.exp(theta[0]))*sp.stats.poisson.pmf(k[1], np.exp(theta[1]))
    return f

def fhn(Y, deltat):
    
    ''' first order euler discretization of the fitzhugh nagumo differential equations (not used) '''
    
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

def apf(obs, time, n_particles, n_gridpoints, A, B, Sigma, x_0):
    '''
    Implements the Auxiliary Particle Filter as described by Whiteley and Johansen
    Algorithm 2 on page 5: http://www.maths.bris.ac.uk/~manpw/apf_chapter.pdf
    
    Input: 
    [obs]           : a time x dimension matrix representing a time series of observed signals
    [time]          : a scalar representing the corresponding time length
    [n_particles]   : a scalar representing the number of particles to use in the simulation
    [n_gridpoints]  : a scalar representing the number of grid points or nodes to use for the quadrature
    [B]             : a 2x2 propagator matrix for the dynamics
    [Sigma]         : a 2x2 covariance matrix
    [Gamma]         : a 2x2 covariance matrix
    [x_0]           : a 2x1 vector of representing the initial value of the signal
    [I_ext]         : a scalar representing input current magnitude
    Output: 
    [W]             : an n_particles x time matrix of weights
    [X]             : an n_particles x time x dimension tensor of trajectories
    [k]             : an n_particles x time matrix of the posterior integral for each particle at each time point
    Average the trajectory tensor (X) across particles to approximate the functional integral. Smooth the
    resulting signal to remove noise.
    
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
    # TO DO: generalize the computation of posterior integral
    XX = np.tile(xt.reshape([1, len(xt)]), [len(xt), 1])
    YY = np.tile(xt.reshape([len(xt), 1]), [1, len(xt)])
    T = sp.linalg.sqrtm(Sigma)

    # sample particles and weights at time 1
    import pdb
    for i in range(0,n_particles):
        X[i,0,:] = np.random.randn(1,dimension)[0] 
        g = make_poisson(obs[0,:])(np.dot(B,X[i,0,:]))
        nu = make_mvn_pdf(x_0, Sigma)(X[i,0,:])
        q = make_mvn_pdf(np.zeros(dimension), proposal_covariance_matrix)(X[i,0,:])
        W[i,0] = g*nu/q

    
    # main loop of program at time > 1
    for t in range(1, time):

        # Update weights and propagate particles based on posterior integral
        for i in range(n_particles):
            # Compute the posterior integral p(y_n | x_{n-1})
            g_int_func = make_poisson(obs[t,:])
            k[i,t] = bivariate_gauss_hermite(xt,wt, np.dot(A,X[i, t-1,:]), T, g_int_func, XX, YY )
            #print "posterior integral: ", k[i,t]
            # Reweight particles
            W[i,t-1] = W[i,t-1]*k[i,t]

        # Resample
        Xprime = np.random.choice(N_particles, n_particles, p = W[:,t-1]/np.sum(W[:,t-1]), replace = True)
        Xtilde = [X[i,t-1,:] for i in Xprime]
        
        # Reset weights and particles
        for i in range(n_particles):
            # Select new particles            
            X[i,t-1,:] = Xtilde[i]
            # Resample particles and reset weights
            X[i,t,:] = np.random.randn(1,dimension)[0] + X[i,t-1,:]
            #print "particles: ", X[i,t,:]
            # Update proposal and target distributions
            g = make_poisson(obs[t])(np.dot(B,X[i,t,:]))
            q = make_mvn_pdf(X[i,t-1,:],np.identity(dimension))(X[i,t,:])
            f = make_mvn_pdf(np.dot(A,X[i,t-1,:]), Sigma)(X[i,t,:])
            # Update weights
            W[i,t] = (g*f)/(k[i,t]*q)

        print "time: ", t
    return W, X, k

