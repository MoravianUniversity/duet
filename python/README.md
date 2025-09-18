DUET Blind Source Separation Library in Python
==============================================

This repository a Python implementation of the DUET (Degenerate Unmixing Estimation Technique) blind source separation algorithm.

This code is based on the original MATLAB implementation by Dr. J. Rickard (<https://github.com/yvesx/casa495>) and the near-direct port to Python (<https://github.com/BrownsugarZeer/BSS_DUET>). It has been improved for performance and usability, running about four times faster than the previous Python version and supporting "online" processing of audio streams.

The duet_base module includes the base parts of the algorithm and provides the general interface for using DUET. It implements all steps except peak-finding since there are several possible algorithms for that step. The base algorithm defines the following parameters:
- `window`: The STFT window to use; larger sizes will result in better frequency resolution but worse time resolution, default is Hamming-256.
- `sample_rate`: The sample rate of the input audio signal in Hz (samples/sec), used in frequency calculations, default is 16000.
- `p`: The symmetric attenuation estimator value weights, default is 1.
- `q`: The delay estimator value weights, default is 0.

It provides the following methods:
- `run(x)`: Run the DUET algorithm on the entire input audio signal, this is the "offline" version of the algorithm.
- `start(x)`: Initialize the online DUET algorithm with the input audio signal `x`. This must be called before any call to `update()`.
- `update(x, return_full=True)`: Update the online DUET algorithm with the new audio signal `x`. This must be called after `start()`.
- `reset()`: Reset the online DUET algorithm, clearing all internal state.

The duet_orig version implements peak-finding using a smoothed 2D histogram and local maxima detection. It has several additional parameters (see the code for details) that control the peak-finding algorithm.

The duet_ms version implements peak-finding using a mean-shift clustering algorithm as per <https://www.sciencedirect.com/science/article/abs/pii/S0165168412000722>. It has several additional parameters (see the code for details) that control the peak-finding algorithm. It has a massive benefit over other peak-finding algorithms in that it does not require the number of sources to be known or estimated ahead of time.

Current features:
- Supports stereo input audio signals (2 channels)
- "Online" processing of audio streams, updating the separation as new audio data arrives, reducing computation time and memory usage
- Support for alternative peak-finding algorithms (2 currently implemented)
- Mean-shift peak-finding does not require the number of sources to be known ahead of time

Future improvements:
- Support for more than 2 input channels (mostly complete, needs demixing calculations)
- "Big delay" support for sources with large time delays between channels (needs testing)
- More peak-finding algorithms
- Selective source removal and reconstruction

Notebooks
---------

The notebooks contains code for testing and demonstrating the library. The CSV files are results from running the notebooks (used to bridge data between the two notebooks).
