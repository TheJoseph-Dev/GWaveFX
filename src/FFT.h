#ifndef FFT_H
#define FFT_H

#include <fftw3.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include <algorithm>

//#define PI 3.141592653

constexpr unsigned LOW_FREQUENCY = 20;
constexpr unsigned HIGH_FREQUENCY = 20000;

constexpr unsigned FFT_BM_WINDOW = 1U; // Apply Blackman Window to input
constexpr unsigned FFT_ABS = 2U; // Absolute value of FFT result
constexpr unsigned FFT_SMOOTH = 4U; // Smooths the result with previous calculated values
constexpr unsigned FFT_CONVERT_TO_DB = 8U; // Converts result to dB and scales/clamps from 0 to 255
constexpr unsigned FFT_CONVOLVE = 16U; // Avarages the FFT output by doing a convolution

class FFT {

	size_t nSamples;
	double* out;
	float* fOut;
	float* inConv; // Used as input to the convolution
	fftw_plan plan;

public:
	double* in;

public:
	FFT(size_t nSamples, unsigned int planFlags = FFTW_ESTIMATE);
	~FFT();

	void Execute(unsigned int flags = 0U);

	inline double* getResult() const { return this->out; };
	inline float* getResultF() const { return this->fOut; };
};

#endif // !FFT_H
