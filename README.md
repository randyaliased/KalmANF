### Frequency trackers based on an adaptive notch filter (ANF).

The frequency trackers in this repo. are specifically based on updating a single parameter in a constrained biquad filter. They are useful for tracking the frequency variation of a high-energy sinusoidal compoenent in any signal. The examples used here are primarily audio signal. The following frequency trackers are implemented: 

1. LMS-ANF: An ANF updated with an LMS algorithm
2. KalmANF: Anf ANF updated with a Kalman filter

The repo. also contains the experiments to regenerate the figures from the paper: R. Ali, T. van Waterschoot, "A frequency tracker based on a Kalman filter update of a single parameter adaptive notch filter" 

