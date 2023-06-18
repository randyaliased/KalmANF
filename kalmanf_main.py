#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 07:24:15 2023

Main script for analysing the performance of Adaptive Notch Filters used in the following work:
R. Ali, T. van Waterschoot, "A frequency tracker based on a Kalman filter update of a single parameter adaptive notch filter", 
Proceedings of the 26th International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, September 2023

@author: randallali
"""


import numpy as np 
from scipy import signal
from matplotlib import pyplot as plt
import adaptive_notch_filters as anf
import soundfile as sf

# 1. Read in the input signal

y, fs = sf.read('data/XC513058_Musician_Wren_Cyphorhinus_arada_short.wav')

L = len(y)    # Total duration of signal (number of samples)
n = np.arange(0,L,1) # range of samples
tt = n/fs   # time range of signal


#%
# 2. Run ANFs

rho = 0.95

f_lms, a_lms, e_lms = anf.lms (y, fs, rho, mu=0.3) # LMS
f_kal, a_kal, e_kal = anf.kalmanf (y, fs, rho, q=8e-5, r=0.01) # KalmANF

#%

# Spectrograms
nfft = 512         # number of points for the FFT 
noverlap = nfft/2  # Spectrogram overlap (make it 50 %)

# Spectrogram of the input signal
f_sg, t_sg, Z_mag = signal.spectrogram(y, fs=fs,nperseg=nfft,window='hann',mode='magnitude',noverlap=noverlap)
Z_dB = 10*np.log10(Z_mag**2) # convert the magnitude to dB
extent = t_sg[0], t_sg[-1], f_sg[0], f_sg[-1]  # this defines the 4 corners of the "image" for the imshow function to plot spectrogram

# Spectrogram of the KalmANF error
f_sg, t_sg, E_mag = signal.spectrogram(e_kal, fs=fs,nperseg=nfft,window='hann',mode='magnitude',noverlap=noverlap)
E_dB = 10*np.log10(E_mag**2) # convert the magnitude to dB

min_dB = -140
max_dB = -40



fig, axes = plt.subplots(3,1)
fig.subplots_adjust(hspace=0.5)  # horizontal spacing

sg = axes[0].imshow(Z_dB, origin='lower',aspect='auto',extent=extent, vmin=min_dB, vmax=max_dB)
axes[0].set_xlim(t_sg[0], t_sg[-1])
axes[0].set_ylim(0, 4000)
axes[0].set_xlabel('Time (s)')
axes[0].set_title('Input Signal')
axes[0].set_ylabel('Frequency (Hz)')
cb = plt.colorbar(sg,ax=[axes[0]])

sg = axes[1].imshow(Z_dB, origin='lower',aspect='auto',extent=extent, vmin=min_dB, vmax=max_dB)
axes[1].plot(tt,f_lms,'m--',linewidth=1.2, label ='LMS-ANF')
axes[1].plot(tt,f_kal,'k--',linewidth=1.2,label ='KalmANF')
axes[1].set_xlim(t_sg[0], t_sg[-1])
axes[1].set_ylim(0, 4000)
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Input Signal + Freq. tracks')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].legend()
cb1 = plt.colorbar(sg,ax=[axes[1]])


sg_e = axes[2].imshow(E_dB, origin='lower',aspect='auto',extent=extent, vmin=min_dB, vmax=max_dB)
axes[2].set_xlim(t_sg[0], t_sg[-1])
axes[2].set_ylim(0, 4000)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Frequency (Hz)')
axes[2].set_title('Filtered Output from KalmANF')
cb2 = plt.colorbar(sg_e,ax=[axes[2]])




#%%



