#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 07:24:15 2023

Main script for analysing the performance of Adaptive Notch Filters

@author: randallali
"""

import numpy as np 
from scipy import signal
from matplotlib import pyplot as plt
import adaptive_notch_filters as anf
import soundfile as sf
import tikzplotlib

# 1. Read in the input signal

y, fs = sf.read('XC513058_Musician_Wren_Cyphorhinus_arada_short.wav')

L = len(y)    # Total duration of signal (number of samples)
n = np.arange(0,L,1) # range of samples
tt = n/fs   # time range of signal


#%
# 2. Run algorithms

rho = 0.95
mu = 0.3
q = 8e-3
r = 1

f_lms, a_lms, e_lms = anf.lms (y, fs, rho, mu) # LMS
f_kal, a_kal, e_kal = anf.kalmanf (y, fs, rho, q, r) # Kalman

sf.write('kalmanf_filtered.wav', e_kal, samplerate=fs)
#%

# Plotting the spectrogram. We use the package signal which has a spectrogram function
nfft = 512         # number of points for the FFT 
noverlap = nfft/2  # Spectrogram overlap (make it 50 %)


#%
# Compute the spectrogram. We set the mode to obtain the magnitude, 
f_sg, t_sg, Z_mag = signal.spectrogram(y, fs=fs,nperseg=nfft,window='hann',mode='magnitude',noverlap=noverlap)
Z_dB = 10*np.log10(Z_mag**2) # convert the magnitude to dB
extent = t_sg[0], t_sg[-1], f_sg[0], f_sg[-1]  # this defines the 4 corners of the "image" for the imshow function to plot spectrogram

# Spectrogram of the error
f_sg, t_sg, E_mag = signal.spectrogram(e_kal, fs=fs,nperseg=nfft,window='hann',mode='magnitude',noverlap=noverlap)
E_dB = 10*np.log10(E_mag**2) # convert the magnitude to dB

min_dB = -130
max_dB = -40

#%
downsample = 100 # Plotting every 10th sample - for saving figure

fig, axes = plt.subplots(3,1)

sg = axes[0].imshow(Z_dB, origin='lower',aspect='auto',extent=extent, vmin=min_dB, vmax=max_dB)
# plt.plot(tt,f_lms,'--',label ='LMS ANF')
# plt.plot(tt,f_kal,'--',label ='KalmANF')
axes[0].set_xlim(t_sg[0], t_sg[-1])
axes[0].set_ylim(0, 4000)
# axes.set_xlabel('Time (s)')
axes[0].set_ylabel('Frequency (Hz)')
cb = plt.colorbar(sg,ax=[axes[0]],location='top')

sg = axes[1].imshow(Z_dB, origin='lower',aspect='auto',extent=extent, vmin=min_dB, vmax=max_dB)
axes[1].plot(tt[::downsample],f_lms[::downsample],'m-',linewidth=1.2, label ='LMS ANF')
axes[1].plot(tt[::downsample],f_kal[::downsample],'k-',linewidth=1.2,label ='KalmANF')
axes[1].set_xlim(t_sg[0], t_sg[-1])
axes[1].set_ylim(0, 4000)
# axes.set_xlabel('Time (s)')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].legend()

sg_e = axes[2].imshow(E_dB, origin='lower',aspect='auto',extent=extent, vmin=min_dB, vmax=max_dB)
axes[2].set_xlim(t_sg[0], t_sg[-1])
axes[2].set_ylim(0, 4000)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Frequency (Hz)')
# cb2 = plt.colorbar(sg,ax=[axes[0]],location='top')

# tikzplotlib.save("./Fig_bird/Fig_bird.tex") 

# # Write Text file with the Simulation parameters used
# with open ('./Fig_bird/KalmANF_Bird_Parameters.txt', 'w') as file:  
#     file.write('Simulation Parameters used to generate Figure')
#     file.write('\n')  
#     file.write('\n') 
#     file.write('Rho (pole radius) = '+str(rho)+'\n')
#     file.write('\n')
#     file.write('LMS Parameters: \n')
#     file.write('mu (step size) = '+str(mu)+'\n')
#     file.write('\n')
#     file.write('KalmANF Parameters: \n')
#     file.write('q (process noise covaraince) = '+str(q)+'\n')
#     file.write('r (measurement noise covariance) = '+str(r)+'\n')


#%%



