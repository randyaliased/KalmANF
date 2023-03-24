#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 07:24:15 2023

Monte Carlo for analysing the performance of Adaptive Notch Filters using simulated data
We will set up a sinusoidal signal + random gaussian noise and average over several runs.
In this script we analyze the performance in terms of SNR and rho

@author: randallali
"""
import sys
sys.path.append('../')
import numpy as np 
from matplotlib import pyplot as plt
import adaptive_notch_filters as anf
from tqdm import tqdm
import tikzplotlib


# 1. Generate the signal + noise

fs = 8000   # sampling frequency (Hz)
T = 4       # Total duration of signal (s)
L = T*fs    # Total duration of signal (number of samples)
n = np.arange(0,L,1) # range of samples
tt = n/fs   # time range of signal

f1 = 868
w1 = 2*np.pi*f1/fs
x = 0.5*np.cos(w1*np.arange(0,L,1))
f_true = np.repeat(f1,len(x))

SNRs = [-5,0, 5, 10, 15] #np.arange(-5,20,5)
N_SNR = len(SNRs)
rho_small = 0.6 # ANF pole radius
rho_large = 0.95

# These paramters were tuned to have same convergence rate for -5 dB 
# LMS parameters
mu = 1e-3


#KalmANF Parameters:
q_large= [1e-4, 4.5e-5, 2.5e-5, 1.9e-5, 1.7e-5] 
q_small= [4e-5, 2.5e-5, 2e-5, 2e-5, 2e-5]
r=10

N_mc = 100 # number of monte carlo runs

# For large values of rho
Mis_LMS_MC_rholarge = np.zeros([N_SNR,L,N_mc]) # misalignments for each SNR and MC run
Mis_KAL_MC_rholarge = np.zeros([N_SNR,L,N_mc])
f_LMS_MC_rholarge = np.zeros([N_SNR,L,N_mc]) # freq estimates
f_KAL_MC_rholarge = np.zeros([N_SNR,L,N_mc])
Mis_LMS_MC_SS_rholarge = np.zeros(N_SNR) # steady state mean of the Misalignment
Mis_KAL_MC_SS_rholarge = np.zeros(N_SNR)

Mis_LMS_MC_rhosmall = np.zeros([N_SNR,L,N_mc]) # misalignments for each SNR and MC run
Mis_KAL_MC_rhosmall = np.zeros([N_SNR,L,N_mc])
f_LMS_MC_rhosmall = np.zeros([N_SNR,L,N_mc]) # freq estimates
f_KAL_MC_rhosmall = np.zeros([N_SNR,L,N_mc])
Mis_LMS_MC_SS_rhosmall = np.zeros(N_SNR) # steady state mean of the Misalignment
Mis_KAL_MC_SS_rhosmall = np.zeros(N_SNR)

for i in range(N_SNR):
    
    SNR = SNRs[i] # desired SNR (dB) (10*np.log10(np.var(x)/np.var(noise)))

    for m in tqdm(range(N_mc)):
        
        # Create new realisation of noise and input signal
        mean = 0
        std = np.sqrt(np.var(x)/(10**(SNR/10)))
        noise = np.random.normal(mean, std, size=L)
        y = x + noise # this is the input signal
    
        # For Large rho
        # Run ANF algorithms - we are only going to overage the estimated frequency, 
        # so we do not care about the other 2 output arguments, but use them for plotting if so desired
        f_LMS_MC_rholarge[i,:,m], _, _ = anf.lms (y, fs, rho_large, mu) # LMS
        f_KAL_MC_rholarge[i,:,m], _, _ = anf.kalmanf (y, fs, rho_large, q_large[i], r) # Kalman
        
        # Compute misalignments
        Mis_LMS_MC_rholarge[i,:,m] = anf.norm_misalignment(f_true, f_LMS_MC_rholarge[i,:,m])
        Mis_KAL_MC_rholarge[i,:,m] = anf.norm_misalignment(f_true, f_KAL_MC_rholarge[i,:,m])
        
        
        # For small rho
        f_LMS_MC_rhosmall[i,:,m], _, _ = anf.lms (y, fs, rho_small, mu) # LMS
        f_KAL_MC_rhosmall[i,:,m], _, _ = anf.kalmanf (y, fs, rho_small, q_small[i], r) # Kalman
        
        Mis_LMS_MC_rhosmall[i,:,m] = anf.norm_misalignment(f_true, f_LMS_MC_rhosmall[i,:,m])
        Mis_KAL_MC_rhosmall[i,:,m] = anf.norm_misalignment(f_true, f_KAL_MC_rhosmall[i,:,m])


    # For Large rho
    # Computing mean
    Mis_LMS_MC_rholarge_mean = np.mean(Mis_LMS_MC_rholarge[i,:,:],axis=1)
    Mis_KAL_MC_rholarge_mean = np.mean(Mis_KAL_MC_rholarge[i,:,:],axis=1)
    
    f_LMS_MC_rholarge_mean = np.mean(f_LMS_MC_rholarge[i,:,:],axis=1)
    f_KAL_MC_rholarge_mean = np.mean(f_KAL_MC_rholarge[i,:,:],axis=1)
    
    # Computing the steady state averages - will compute the average of the latter half of the convergence
    Mis_LMS_MC_SS_rholarge[i] = np.mean(Mis_LMS_MC_rholarge_mean[L//2:])
    Mis_KAL_MC_SS_rholarge[i] = np.mean(Mis_KAL_MC_rholarge_mean[L//2:])


    # For small rho
    Mis_LMS_MC_rhosmall_mean = np.mean(Mis_LMS_MC_rhosmall[i,:,:],axis=1)
    Mis_KAL_MC_rhosmall_mean = np.mean(Mis_KAL_MC_rhosmall[i,:,:],axis=1)
    
    f_LMS_MC_rhosmall_mean = np.mean(f_LMS_MC_rhosmall[i,:,:],axis=1)
    f_KAL_MC_rhosmall_mean = np.mean(f_KAL_MC_rhosmall[i,:,:],axis=1)
    
    # Computing the steady state averages - will compute the average of the latter half of the convergence
    Mis_LMS_MC_SS_rhosmall[i] = np.mean(Mis_LMS_MC_rhosmall_mean[L//2:])
    Mis_KAL_MC_SS_rhosmall[i] = np.mean(Mis_KAL_MC_rhosmall_mean[L//2:])



#%
#% PLOTTING 

fig, ax = plt.subplots()
plt.plot(SNRs,Mis_LMS_MC_SS_rholarge, '-o',label='LMS ANF, rho = '+str(rho_large))
plt.plot(SNRs,Mis_KAL_MC_SS_rholarge, '-x',label='KalmANF, rho = '+str(rho_large))
plt.plot(SNRs,Mis_LMS_MC_SS_rhosmall, '--o',label='LMS ANF, rho = '+str(rho_small))
plt.plot(SNRs,Mis_KAL_MC_SS_rhosmall, '--x',label='KalmANF, rho = '+str(rho_small))
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Averaged Steady State Normalised Misalignment (dB)')
ax.grid()
ax.legend()


# downsample = 10 # Plotting every 10th sample - for saving figure
# fig, ax = plt.subplots()
# plt.plot(tt[::downsample],Mis_LMS_MC_rholarge_mean[::downsample], label='LMS ANF')
# plt.plot(tt[::downsample],Mis_KAL_MC_rholarge_mean[::downsample], label='KalmANF')
# plt.plot(tt[::downsample],Mis_LMS_MC_rhosmall_mean[::downsample], label='LMS ANF')
# plt.plot(tt[::downsample],Mis_KAL_MC_rhosmall_mean[::downsample], label='KalmANF')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Normalised Misalignment (dB)')
# ax.grid()
# ax.legend()

# tikzplotlib.save("../results/Fig_SNR/Norm_Mis_FigSNR.tex")

# Uncomment to see the actual frequency
# fig, ax = plt.subplots()
# plt.plot(tt,f_true,'-', label='True')
# plt.plot(tt,f_LMS_MC_mean,label ='LMS ANF')
# plt.plot(tt,f_KAL_MC_mean,label ='KalmANF')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Frequency (Hz)')
# ax.legend()



# #%
# # Write Text file with the Simulation parameters used
# with open ('../results/Fig_SNR/KalmANF_MonteCarlo_Sim_Parameters.txt', 'w') as file:  
#     file.write('Simulation Parameters used to generate Figure')
#     file.write('\n')  
#     file.write('\n') 
#     file.write('Number of Monte Carlo Runs = '+str(N_mc)+' \n')
#     file.write('SNR = '+str(SNRs)+'dB \n')
#     file.write('Rho Large (pole radius) = '+str(rho_large)+'\n')
#     file.write('Rho Small (pole radius) = '+str(rho_small)+'\n')
#     file.write('\n')
#     file.write('LMS Parameters: \n')
#     file.write('mu (step size) = '+str(mu)+'\n')
#     file.write('\n')
#     file.write('KalmANF Parameters: \n')
#     file.write('q_large (process noise covaraince) = '+str(q_large)+'\n')
#     file.write('q_small (process noise covaraince) = '+str(q_small)+'\n')
#     file.write('r (measurement noise covariance) = '+str(r)+'\n')
