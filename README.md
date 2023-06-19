### KalmANF: A frequency tracker based on a Kalman filter update of a single parameter adaptive notch filter 
 

A particular class of frequency trackers are based on updating a single parameter in a constrained biquad filter configured as a notch filter.  Such an update can be done using a least mean square (LMS) algorithm, or a Kalman filter update, which turns out to be a regularized normalized least mean square algorithm with an optimal regularization parameter, the details of which are discussed in the following paper:

R. Ali, T. van Waterschoot, "[A frequency tracker based on a Kalman filter update of a single parameter adaptive notch filter](https://ftp.esat.kuleuven.be/pub/SISTA/rali/Reports/23-57.pdf)", accepted for Proceedings of the 26th International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, September 2023.

The scripts/directories in this repo. are as follows:

1.  adaptive_notch_filters. py | 
    *This script contains the functions of the ANFs: (i) An ANF updated with an LMS algorithm (LMS-ANF) and (ii) An ANF updated with a Kalman filter (KalMANF)*.

2. kalmanf_main.py |  *The main analysis script. You can load any audio file and obeserve the results from the LMS-ANF and KalMAF. Currently audio files from the /data folder are used*.

3. /simulations | *This directory contains the various scripts to regenerate the figures from the aforementioned paper. The results are generated in the /results folder*.

4. /jupyter-notebook | *This directory contains an interactive jupyter notebook where sounds can be recorded from your machine and readily analyzed with both the LMS-ANF and KalmANF. This has been a great practical teaching aid when getting into topics of adaptive filters and Kalman filters in particular*.



