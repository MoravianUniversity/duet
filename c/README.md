DUET Blind Source Separation Library for C/C++
==============================================

This repository contains a C/C++ implementation of the DUET (Degenerate Unmixing Estimation Technique) blind source separation algorithm along with Python bindings via Cython.

This is designed to be used on embedded systems but also compiled as a shared library for use on desktop systems for comparison and testing. The implementation is based on the Python implementation in the `python` directory but has a few differences:
- "Online" support only, meaning it processes audio streams as they arrive rather than waiting for the entire signal to be available. This reduces memory usage and latency.
- Mean-shift peak-finding algorithm support only, as it is the most flexible and robust peak-finding algorithm found.
- Mean-shift algorithm has been altered slightly for performance and to reduce memory usage.
- None of the mean-shift parameters are adjustable at runtime.
- Many math operations have been optimized for performance, using good approximations where possible.
- No support for "big delay" mode.
- Only partial support for more than 2 input channels.
- For digital signal processing, the [ESP-DSP](https://github.com/espressif/esp-dsp) library is used, this includes Fourier transforms and signal filtering/decimation.

Due to all of this, the C/C++ implementation will not generate exactly the same results as the Python implementation, but they should be close. It is several times faster than the Python implementation (likely about 4-8 times faster).

Extra Files
-----------

- `DUET C Testing.ipynb` - Jupyter notebook for testing the C library and comparing it to the Python implementation.
- `CMakeLists.txt` - CMake build file for building the C/C++ library on desktop systems. Currently only tested on macOS.
- `esp-idf-dummy` - Dummy header files from the ESP-IDF framework to allow compilation on non-ESP32 platforms.
- `duet-lib.ipynb` - Jupyter notebook for testing the Cython bindings to the C library.

TODO: add setup.py or similar to build the Cython bindings once working.

To compile on desktop systems, you will need to have CMake and the ESP-DSP library installed. You can then run the following commands in the `c` directory:

```bash
git clone https://github.com/espressif/esp-dsp.git  # TODO: needs some modifications to work properly
mkdir build
cd build
cmake ..
make
```

TODO: add instructions for building on ESP32.
