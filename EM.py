import os
import glob
import csv
import time
from   pylab import *
from   datetime import datetime
import numpy as np
import sys
import math
from   operator import truediv
import _pickle as pickle
from   random import *
import scipy.stats
import pypolyagamma
import multiprocessing as mp
from scipy import linalg
import matplotlib.pyplot as plt


def TransformToProb(meanv, sigma2, mu):
    '''
    Transform the computed mean, sigma2 into probability
    '''
    #compute upper and lower bounds of conf intervals by simulation
    NUM_SAMPS = 10000

    np.random.seed(0) # start at same random number each time
    T       = len(meanv)
    p       = np.zeros(T)
    pll     = np.zeros(T)
    pul     = np.zeros(T)
    pmode   = np.zeros(T)
    sigma   = np.sqrt(sigma2)

    for t in range(T):
        s         = np.random.normal(meanv[t], sigma[t], NUM_SAMPS)
        ps        = list(map(truediv, np.exp(s + mu), (1.0+np.exp(s + mu))))
        pmode[t]  = np.exp(meanv[t] + mu)/(1.0+np.exp(meanv[t] + mu))
        p[t]      = np.mean(ps)
        pll[t],pul[t] = np.percentile(ps,[2.5,97.5])
    return (pmode, p, pll, pul)

def NewtonSolve(x_prior, sigma_prior, N, binomialN, mu):
    '''
    Solve for posterior mode using Newton's method
    '''
    xp = x_prior
    sp = sigma_prior
    it = xp + sp*(N - binomialN*np.exp(mu+xp)/(1.0 + np.exp(mu+xp)))     #starting iteration
    it = 15

    for i in range(30): 
        g     = xp + sp*(N - binomialN*np.exp(mu+it)/(1.0+np.exp(mu+it))) - it;
        gprime = -binomialN*sp*np.exp(mu+it)/(1.0+np.exp(mu+it))**2.0 - 1.0   
        x = it  - g/gprime 

        #print g, gprime, x

        if np.abs(x-it)<1e-10:
            return x
        it = x

    #if no value found try different ICs: needed if there are consec same values
    it = -1
    for i in range(30): 
        g     = xp + sp*(N - binomialN*np.exp(mu+it)/(1.0+np.exp(mu+it))) - it
        gprime = -binomialN*sp*np.exp(mu+it)/(1.0+np.exp(mu+it))**2.0 - 1.0
        x = it  - g/gprime 

        if np.abs(x-it)<1e-10:
            return x
        it = x

    #if no value found try different ICs
    it = 1
    for i in range(30): 
        g     = xp + sp*(N - binomialN*np.exp(mu+it)/(1+np.exp(mu+it))) - it
        gprime = -binomialN*sp*np.exp(mu+it)/(1+np.exp(mu+it))**2 - 1.0
        x = it  - g/gprime 

        if np.abs(x-it)<1e-10:
            return x
        it = x

def FwdFilterEM(y,delta,x_init,sigma2_init,sigma2e, mu, binomialN):
    '''
    EM step 1: The forward nonlinear recursive filter
    '''
    T = y.shape[1]
    # Data structures
    x_prior = zeros(T+1) # xk|k-1
    x_post  = zeros(T+1) # xk-1|k-1

    next_pred_error = zeros(T+1)

    sigma2_prior = zeros(T+1) # sigma2k|k-1
    sigma2_post  = zeros(T+1) # sigma2k-1|k-1

    # FORWARD FILTER
    x_post[0]      = x_init
    sigma2_post[0] = sigma2_init 

    for t in range(1,T+1):
        #print t

        x_prior[t]      = x_post[t-1]
        sigma2_prior[t] = sigma2_post[t-1] + sigma2e
        
        N = np.sum(y[:,t-1])
        x_post[t]  = NewtonSolve(x_prior[t],sigma2_prior[t],N,binomialN[t-1],mu)

        pt = exp(mu+x_post[t])/(1.0+exp(mu+x_post[t]))
        sigma2_post[t] = 1.0 / ( 1.0/sigma2_prior[t] + binomialN[t-1]*pt*(1-pt))

    ape = 0#next_pred_error.mean()

    return x_prior,x_post,sigma2_prior,sigma2_post, ape

def BackwardFilter(x_post, x_prior, sigma2_post, sigma2_prior):
    '''
    EM Step 1: Fixed Interval Smoothing Algorithm
    '''
    T = len(x_post)
    # Initial conditions
    x_T               = zeros(T)  
    x_T[T-1]          = x_post[T-1]
    sigma2_T          = zeros(T)
    sigma2_T[T-1]     = sigma2_post[T-1]
    A = np.zeros(T)

    for t in range(T-2,0,-1):
        A[t]            = sigma2_post[t]/sigma2_prior[t+1]
        x_T[t]          = x_post[t] + np.dot(A[t],x_T[t+1] - x_prior[t+1])
        Asq             = np.dot(A[t],A[t])
        diff_v          = sigma2_T[t+1] - sigma2_prior[t+1]
        sigma2_T[t]     = sigma2_post[t] + np.dot(Asq, diff_v)

    return x_T,sigma2_T,A

def MSTEP(xnew, signewsq, A):
    '''
    M step of EM
    '''
   
    T          = len(xnew)
    xnewt      = xnew[2:T]
    xnewtm1    = xnew[1:T-1]
    signewsqt  = signewsq[2:T]
    A          = A[1:T-1]
    covcalc    = np.multiply(signewsqt,A)

    term1      = np.dot(xnewt,xnewt) + np.sum(signewsqt)
    term2      = np.sum(covcalc) + np.dot(xnewt,xnewtm1)

    term3      = 2*xnew[1]*xnew[1] + 2*signewsq[1]
    term4      = xnew[T-1]**2 + signewsq[T-1]

    newsigsq   = (2*(term1-term2)+term3-term4)/T

    return newsigsq

def EM(xx, mu, sigma2e, x_init, sigma_init, binomialN):
    '''
    xx : Neuron Spike Data
    '''
    num_its         = range(0,3000)   
    savesigma2_e    = np.zeros(len(num_its)+1)
    savesigma2_e[0] = sigma2e

    #run though until convergence
    its      = 0
    diff_its = 1
    max_its  = 3000

    while diff_its>0.00001 and its <= max_its:
        its +=  1
        x_prior,x_post,sigma2_prior,sigma2_post, ape = FwdFilterEM(xx, 1, x_init, sigma_init, sigma2e, mu, binomialN) 


        x_T,sigma2_T,A   = BackwardFilter(x_post,x_prior,sigma2_post, sigma2_prior)

        # x_T[0]     = 0               
        # sigma2_T[0] = sigma2e

        sigma2e   = MSTEP(x_T, sigma2_T, A)  

        savesigma2_e[its+1]  = sigma2e       
        diff_its             = abs(savesigma2_e[its+1]-savesigma2_e[its])

        x_init     = x_T[0]           
        sigma_init = sigma2_T[0]


    if its == max_its:
        converge_flag = 1
        print ('Did not converge in 3000 iterations')
    else:
        converge_flag = 0
        print ()
        print ('Converged after ' + str(its) + ' iterations')
        print ('sigma2e is ', sigma2e)

    # x_post      = x_post[1:]
    # sigma2_post = sigma2_post[1:]
    #return x_T[1:],sigma2_T[1:],sigma2e,sigma_init,converge_flag
    return x_post,sigma2_post,sigma2e,sigma_init,converge_flag

def RunEM(values, binomialN, sigma2e = 0.5**2):
    '''
    Run the EM algorithm and return probability with confidence bands
    '''
    t0 = time.time()
    startflag  = 0
    # sigma2e    = 0.5**2 #start guess


    sigma_init = sigma2e
    x_init = 0
    mu = 0
    
    print ('initial sigma2e is', sigma2e)
    x_post,sigma2_post,sigma2e,sigma_init,converge_flag =  EM(values, mu, sigma2e, x_init, sigma_init, binomialN)
    pmode, p, pll, pul = TransformToProb(x_post[1:], sigma2_post[1:], mu)
    print ('runtime: %s seconds' % (time.time()-t0))
    return pmode, p, pll, pul,sigma2e, x_post, sigma2_post


def FwdFilterEM1(y, w, x_init,sigma2_init,sigma2e, binomialN):
    '''
    EM step 1: The forward nonlinear recursive filter
    Inputs:
    y - binomial observations
    w - Samples from Polya-gamma Distribution
    x_init, w_init: initial values for x and w
    binomialN - vector of shots attempted
    '''
    K = y.shape[0]

    x_prior = np.zeros(K+1) # xk|k-1
    x_post  = np.zeros(K+1) # xk-1|k-1


    sigma2_prior = np.zeros(K+1) # sigma2k|k-1
    sigma2_post  = np.zeros(K+1) # sigma2k-1|k-1

    # FORWARD FILTER
    x_post[0]      = x_init
    sigma2_post[0] = sigma2_init 

    for t in range(1,K+1):

        x_prior[t]      = x_post[t-1]
        sigma2_prior[t] = sigma2_post[t-1] + sigma2e
        
        x_post[t]  = (x_prior[t] + sigma2_prior[t]*(y[t-1]-binomialN[t-1]/2.))/ (1. + w[t]*sigma2_prior[t])
        sigma2_post[t] = sigma2_prior[t] / ( 1.0 + w[t]*sigma2_prior[t])

    return x_prior[1:],x_post[1:],sigma2_prior[1:],sigma2_post[1:]


# Function for Gibbs Sampling using filter backwards
def Gibbs_Sampler2(N, burnin, sigma2e, y, x, binomialN,thin = 0):
    '''
    N: Number of Samples
    thin: thinning parameter
    burnin: Number of samples to burnin
    x_init, w_init: initial values for x and w
    binomialN - vector of shots attempted
    '''
    K = y.shape[0]
    shape_params = binomialN.astype(double)
#     nthreads = pypolyagamma.get_omp_num_threads()
#     seeds = np.random.randint(2**16, size=nthreads)
#     ppgs = [pypolyagamma.PyPolyaGamma(seed) for seed in seeds]
    ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2**16))
    # actual number of samples needed with thining and burin-in
    if(thin != 0):
        N_s = N * thin + burnin
    else:
        N_s = N + burnin
    samples = np.empty((N_s,K+1))
    w = np.zeros(K+1)
    for i in range(N_s):
        if(i % 1000 == 0):
            print(i, end=" ")
            # print i,
            sys.stdout.flush()
        #sample the conditional distributions x, w
        #w = pg.polya_gamma(a=shape_params, c=abs(x))
        ppg.pgdrawv(shape_params, abs(x), w)
        #pypolyagamma.pgdrawvpar(ppgs, shape_params, abs(x), w)

        x_prior, x_post,sigma2_prior, sigma2_post = FwdFilterEM1(y, w, x_init=0,sigma2_init=0,sigma2e=sigma2e, binomialN=binomialN)
        mean = x_post[-1]
        var = sigma2_post[-1]
        x[K] = np.random.normal(loc=mean, scale = np.sqrt(var))
        for k in range(K-1):
            # update equations
            x_star_post = x_post[K-k-2] + (sigma2_post[K-k-2]/(sigma2e+sigma2_post[K-k-2]))*(x[K-k] - x_post[K-k-2])
            sigma2_star_post = 1./(1./sigma2e+1./sigma2_post[K-k-2])
            
            # Draw sample for x
            x[K-k-1] = np.random.normal(loc=x_star_post, scale = np.sqrt(sigma2_star_post))
        samples[i,:] = x
        
    if(thin == 0):
        return samples[burnin:,:]
    else:
        return samples[burnin:N_s:thin,:]

def BayesianEM(N_samples, burnin, sigma2e, y, binomialN, x_0 = None, thin=0, max_iter = 30, sampler = Gibbs_Sampler2):
    sigmas = []
    sigmas.append(sigma2e)
    it = 0
    diff = 1
    if x_0 is None:
        x_0 = np.zeros(y.shape[0]+1)
    

    while diff > 1e-6 and it <= max_iter:
        print(it, end=" ")
        # print it,
        sys.stdout.flush()
        x = sampler(N_samples,burnin, sigma2e, y, x_0, binomialN, thin)
        x_0 = np.mean(x, axis = 0)

        x_1 = np.roll(x,1)
        x_1[:,0] = 0

        sigma2e = np.mean((x-x_1)**2)
        sigmas.append(sigma2e)

        # diff = abs(sigmas[it+1]-sigmas[it])
        # print (diff, sigma2e)
        it +=  1

    if it >= max_iter:
        converge_flag = 1
        print ('Did not converge in %s iterations' % max_iter)
        print ('sigma2e is ', sigma2e)
    else:
        converge_flag = 0
        print ()
        print ('Converged after %d iterations' % (it))
        print ('sigma2e is ', sigma2e)
    return x, sigmas, converge_flag