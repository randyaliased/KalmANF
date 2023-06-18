#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 07:25:00 2023

@author: randallali

See: R. Ali, T. van Waterschoot, "A frequency tracker based on a Kalman filter update of a single parameter adaptive notch filter", 
     Proceedings of the 26th International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, September 2023

This script contains the function definitions for:
    1. LMS adaptive notch filter
    2. Proposed Kalman adaptive notch filter (KalmANF)
    3. Generation of a few test signals
    4. Evaluation Metrics (Normalised Misalignment)
"""



import numpy as np


def lms (y, fs, rho, mu):
    '''
    Least Mean Square (LMS) Adaptive Notch Filter
    James M. Kates, “Feedback Cancellation in Hearing Aids: Results from a Computer Simulation,” IEEE Transactions on Signal Processing, vol. 39, no. 3, pp. 553–562, 1991.
    
    Input arguments:
        y       - input data vector (Lx1)
        fs      - sampling frequency (Hz)
        rho     - pole raduis (value between 0 and 1, with values closer to 1 creating a narrower notch)
        mu      - Step size parameter for LMS
    
    Returns:
        f_lms   - Estimated frequency over time (Lx1)
        a_lms   - Estimated filter coefficient over time (Lx1)
        e_lms   - Outpupt from notch filter (Lx1)
        
    '''  

    s_lms = np.zeros(len(y)) # intermediate variable of ANF
    e_lms = np.zeros(len(y)) # output of ANF
    a_lms = np.zeros(len(y)) # coefficient to be updated
    f_lms = np.zeros(len(y)) # frequency to be tracked
    
    for n in np.arange(2,len(y),1): # start loop from two samples ahead because we need samples at m-1 and m-2 
    
        s_lms[n] = y[n] + rho*a_lms[n-1]*s_lms[n-1] - (rho**2)*s_lms[n-2]
        e_lms[n] = s_lms[n] - a_lms[n-1]*s_lms[n-1] + s_lms[n-2]
        a_lms[n] = a_lms[n-1] + 2*mu*e_lms[n]*s_lms[n-1]
    
        if (a_lms[n] > 2) or (a_lms[n] < -2):
            print('a = '+str(a_lms[n])+ ' is out of range, resetting to 0')
            a_lms[n] = 0 # reset coefficient if a is out of range to compute acos

        omega_hat_lms = np.arccos(a_lms[n]/2)
        f_lms[n] = (omega_hat_lms*fs)/(2*np.pi) # estimated frequency
        
    return f_lms, a_lms, e_lms
    

def kalmanf (y, fs, rho, q, r):
    
    '''
    Kalman-Based Adaptive Notch Filter (KalmANF)
    

    Input arguments:
        y       - input data vector (Lx1)
        fs      - sampling frequency (Hz)
        rho     - pole raduis (value between 0 and 1, with values closer to 1 creating a narrower notch)
        q       - Covariance of process noise
        r       - Covariance of measurement noise
        
    Returns:
        f_kal   - Estimated frequency over time (Lx1)
        a_kal   - Estimated filter coefficient over time (Lx1)
        e_kal   - Output from notch filter (Lx1)
        
    '''  


    s_kal = np.zeros(len(y)) # intermediate variable of ANF
    e_kal = np.zeros(len(y)) # output of ANF
    a_kal = np.zeros(len(y)) # coefficient to be updated
    f_kal = np.zeros(len(y)) # frequency to be tracked
    
    p_cov = 0 # initialise covariance of the error
    K = np.zeros(len(y)) # Kalman gain
        
    for n in np.arange(2,len(y),1): # start loop from two samples ahead because we need samples at m-1 and m-2 
        
        # Prediction
        # a(n|n-1) is simply a(n-1) since the state transition matrix is an identiy
        p_cov = p_cov + q; # update covariance of prediction error
        
        
        # Estimation
        s_kal[n] = y[n] + rho*s_kal[n-1]*a_kal[n-1] - (rho**2)*s_kal[n-2] # Define s_kal from data
        K[n] = (s_kal[n-1])/( (s_kal[n-1]**2) + r/p_cov )
        e_kal[n] = s_kal[n] - s_kal[n-1]*a_kal[n-1] + s_kal[n-2] 
        a_kal[n] = a_kal[n-1] + K[n]*e_kal[n]
    
        # Update covariance of error
        p_cov = (1 - K[n]*s_kal[n-1])*p_cov
        
        # Compute frequency
        if (a_kal[n] > 2) or (a_kal[n] < -2):
            print('a = '+str(a_kal[n])+ ' is out of range, resetting to 0')
            a_kal[n] = 0 # reset coefficient if a is out of range to compute acos
        
        omega_hat_kal = np.arccos(a_kal[n]/2)
        f_kal[n] = (omega_hat_kal*fs)/(2*np.pi) # estimated frequency
    
    return f_kal, a_kal, e_kal


def norm_misalignment(f_true, f_est):
    '''
    Normalised Misalignment between the true and estimated parameters
    
    Norm Misalignment (dB) = 20*log10(|f_true-f_est|/f_true)
    
    Input arguments:
        f_true  - True frequency variation over time (Lx1)
        f_est   - Estimated frequency over time (Lx1)
 
    Returns:
        norm_mis   - Normalised Misalignment (dB)
        
    '''  

    norm_mis = 20*np.log10(np.abs(f_true-f_est)/(f_true)) 
    
    return norm_mis


def perc_misalignment(f_true, f_est):
    '''
    Misalignment between the true and estimated parameters in terms of percentage from the true value
    
    Norm Misalignment (dB) = (|f_true-f_est|/f_true)*100
    
    Input arguments:
        f_true  - True frequency variation over time (Lx1)
        f_est   - Estimated frequency over time (Lx1)
 
    Returns:
        perc_mis   - Normalised Misalignment (dB)
        
    '''  

    perc_mis = (np.abs(f_true-f_est)/(f_true))*100 
    
    return perc_mis


def chirp(f1, f2, fs, T, A):
    '''
    Generate an exponential chirp signal (exponential sine sweep)
    
    Input arguments:
        f1      - Start frequency (Hz)
        f2      - End frequency (Hz)
        fs      - Sampling frequency (Hz)
        T       - Total time for chirp (s)
        A       - Amplitude of chirp
        
    Returns:
        x       -  Sweep signal
        f_chirp  -  Freuqency variation of signal over time (Hz)
        
    '''  
    
    L = T*fs    # Total duration of signal (number of samples)
    n = np.arange(0,L,1) # range of samples

    # starting and ending angular freq. for chirp
    w1 = 2*np.pi*f1/fs
    w2 = 2*np.pi*f2/fs
    beta = np.log(w2/w1)/(L-1)

    x = A*np.sin( (w1/beta)*(np.exp(beta*n) - 1 ) ) # generate chirp
    
    w_chirp = w1*np.exp(beta*n)
    f_chirp = (w_chirp*fs)/(2*np.pi)
    
    return x, f_chirp


def two_sines(f1, f2, fs, T, A):
    '''
    Generate a signal with sine @f1 for duration T/2 followed by another sine @ f2 for remaining duration up to T
    
    Input arguments:
        f1      - Frequency of first sine (Hz)
        f2      - Frequency of second sine (Hz)
        fs      - Sampling frequency (Hz)
        T       - Total time for signal (s)
        A       - Amplitude of sines (both will have same amplitude)
        
    Returns:
        x          -  Multi-sine signal
        f_sine     -  Freuqency variation of signal over time (Hz)
        
    '''  
    
    L = T*fs    # Total duration of signal (number of samples)

    # Angular frequency
    w1 = 2*np.pi*f1/fs
    w2 = 2*np.pi*f2/fs
    
    x1 = A*np.sin(w1*np.arange(0,L//2,1))
    x2 = A*np.sin(w2*np.arange(L//2,L,1))
    
    f1_sine = np.repeat(f1,len(x1))
    f2_sine = np.repeat(f2,len(x2))

    f_sine = np.concatenate([f1_sine,f2_sine])
    x = np.concatenate([x1,x2])
    
    return x, f_sine
    
    
    
    
    
    
    


