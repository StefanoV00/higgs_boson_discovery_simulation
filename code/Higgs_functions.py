# -*- coding: utf-8 -*-
"""
@author: Imperial College London
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

np.random.seed(1)
N_b = 10e5 # Number of background events, used in generation and in fit.
b_tau = 30. # Spoiler.

def generate_data(n_signals = 400):
    vals = []
    vals += generate_signal( n_signals, 125., 1.5)
    vals += generate_background( N_b, b_tau)
    return vals

def generate_signal(N, mu, sig):
    return np.random.normal(loc = mu, scale = sig, size = N).tolist()

def generate_background(N, tau):
    return np.random.exponential(scale = tau, size = int(N)).tolist()

def get_B_chi(vals, mass_range, nbins, A, lamb):
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_B_expectation(bin_edges + half_bin_width, A, lamb)
    chi = 0 
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator  
    return chi/float(nbins-2) # B has 2 parameters.

def get_B_expectation(xs, A, lamb):
    return [A*np.exp(-x/lamb) for x in xs]

def signal_gaus(x, mu, sig, signal_amp):
    return signal_amp/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def get_SB_expectation(xs, A, lamb, mu, sig, signal_amp):
    ys = []
    for x in xs:
        ys.append(A*np.exp(-x/lamb) + signal_gaus(x, mu, sig, signal_amp))
    return ys

##############################################################################
#DEFINE THE EXPONENTIAL
def expo (x,A,l):
    return A*np.exp(-x/l)

def signal_gaus(x, mu, sig, signal_amp):
    return signal_amp/(np.sqrt(2.*np.pi)*sig)*\
        np.exp(-np.power((x - mu)/sig, 2.)/2) #JUST HIGGS
        
def signal (x,E,l,G,m,s):
    return expo(x,E,l)+signal_gaus(x,m,s,G) #BACKGROUND+HIGGS