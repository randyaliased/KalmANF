### Frequency trackers based on an adaptive notch filter (ANF).

The frequency trackers in this repo. are specifically based on updating a single parameter in a constrained biquad filter. They are useful for tracking the frequency variation of a high-energy sinusoidal compoenent in any signal. The examples used here are primarily audio signals. The following frequency trackers are implemented: 

1. LMS-ANF: An ANF updated with an LMS algorithm
2. KalmANF: An ANF updated with a Kalman filter

Details of the frequency trackers can be found in the following paper:

R. Ali, T. van Waterschoot, "A frequency tracker based on a Kalman filter update of a single parameter adaptive notch filter", Proceedings of the 26th International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, September 2023. [Archived](https://ftp.esat.kuleuven.be/pub/SISTA/rali/Reports/23-57.pdf)

The repo. also contains the experiments to regenerate the figures from this paper.

An interactive jupyter notebook is also included, where sounds can be recorded from your machine and readily analyzed with both the LMS-ANF and KalmANF. This has been a great practical teaching aid when getting into topics of adaptive filters and Kalman filters in particular. 


