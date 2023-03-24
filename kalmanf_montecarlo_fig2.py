#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 07:24:15 2023

Monte Carlo for analysing the performance of Adaptive Notch Filters using simulated data
We will set up a sinusoidal signal + random gaussian noise and average over several runs.

@author: randallali
"""

import numpy as np 
from scipy import signal
from matplotlib import pyplot as plt
import adaptive_notch_filters as anf
import soundfile as sf
from tqdm import tqdm
import tikzplotlib


# 1. Generate the signal + noise

fs = 8000   # sampling frequency (Hz)
T = 4       # Total duration of signal (s)
L = T*fs    # Total duration of signal (number of samples)
n = np.arange(0,L,1) # range of samples
tt = n/fs   # time range of signal
SNR = 2 # desired SNR (dB) (10*np.log10(np.var(x)/np.var(noise)))

rho = 0.7 # ANF pole radius

# LMS parameters
mu = 1e-3

#KalmANF Parameters:
q=8e-5
r=10
    
# x, f_true = anf.chirp(100, 2000, fs, T, A=0.5)
x, f_true = anf.two_sines(1500, 500, fs, T, A=0.5)

N_mc = 100 # number of monte carlo runs
Mis_LMS_MC = np.zeros([L,N_mc]) # misalignments for each MC run
Mis_KAL_MC = np.zeros([L,N_mc])

f_LMS_MC = np.zeros([L,N_mc]) # frequency estimates for each MC run
f_KAL_MC = np.zeros([L,N_mc])

for m in tqdm(range(N_mc)):
    
    # Create new realisation of noise and input signal
    mean = 0
    std = np.sqrt(np.var(x)/(10**(SNR/10)))
    noise = np.random.normal(mean, std, size=L)
    y = x + noise # this is the input signal

    # Run ANF algorithms - we are only going to overage the estimated frequency, 
    # so we do not care about the other 2 output arguments, but use them for plotting if so desired
    f_LMS_MC[:,m], a_lms, e_lms = anf.lms (y, fs, rho, mu) # LMS
    f_KAL_MC[:,m], a_kal, e_kal = anf.kalmanf (y, fs, rho, q, r) # Kalman
    
    # Compute misalignments
    Mis_LMS_MC[:,m] = anf.norm_misalignment(f_true, f_LMS_MC[:,m])
    Mis_KAL_MC[:,m] = anf.norm_misalignment(f_true, f_KAL_MC[:,m])
    

#%
# Computing mean
Mis_LMS_MC_mean = np.mean(Mis_LMS_MC,axis=1)
Mis_KAL_MC_mean = np.mean(Mis_KAL_MC,axis=1)

f_LMS_MC_mean = np.mean(f_LMS_MC,axis=1)
f_KAL_MC_mean = np.mean(f_KAL_MC,axis=1)


#%
#% PLOTTING Misalignment
downsample = 10 # Plotting every 10th sample - for saving figure

fig, ax = plt.subplots()
plt.plot(tt[::downsample],Mis_LMS_MC_mean[::downsample], label='LMS ANF')
plt.plot(tt[::downsample],Mis_KAL_MC_mean[::downsample], label='KalmANF')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalised Misalignment (dB)')
ax.grid()
ax.legend()

# tikzplotlib.save("./Fig2/Norm_Mis_Fig2.tex")
# plt.savefig('./Fig2/Norm_Mis_Fig2.pdf') 

# # Uncomment to see the actual frequency
# # fig, ax = plt.subplots()
# # plt.plot(tt,f_true,'-', label='True')
# # plt.plot(tt,f_LMS_MC_mean,label ='LMS ANF')
# # plt.plot(tt,f_KAL_MC_mean,label ='KalmANF')
# # ax.set_xlabel('Time (s)')
# # ax.set_ylabel('Frequency (Hz)')
# # ax.legend()



#%
# Write Text file with the Simulation parameters used
with open ('./Fig2/KalmANF_MonteCarlo_Sim_Parameters.txt', 'w') as file:  
    file.write('Simulation Parameters used to generate Figure')
    file.write('\n')  
    file.write('\n') 
    file.write('Number of Monte Carlo Runs = '+str(N_mc)+' \n')
    file.write('SNR = '+str(SNR)+'dB \n')
    file.write('Rho (pole radius) = '+str(rho)+'\n')
    file.write('\n')
    file.write('LMS Parameters: \n')
    file.write('mu (step size) = '+str(mu)+'\n')
    file.write('\n')
    file.write('KalmANF Parameters: \n')
    file.write('q (process noise covaraince) = '+str(q)+'\n')
    file.write('r (measurement noise covariance) = '+str(r)+'\n')
