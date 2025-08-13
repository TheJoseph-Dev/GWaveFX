#include "FFT.h"

template<typename T>
void fftConvolution(T* in, size_t n, double* kernel, int kSize, T* out) {
	
	double kernelSum = 0;
	for (int k = 0; k < kSize; k++) kernelSum += kernel[k];

	for (size_t i = 0; i < n; i++) {
		double v = 0;
		for (int k = -kSize/2; k <= kSize/2; k++)
			v += (i+k > 0 && i+k < n ? in[i+k] : 0) * (kernel[k+kSize/2]/kernelSum);
		out[i] = v;
	}
}

double fftWindow(double s, double N) {
	//fprintf(stdout , "s: %.2lf - N: %.2lf\n", s, N);
	return 0.42 - 0.5 * cos(2.0 * M_PI * s / (N-1)) + 0.08 * cos(4.0 * M_PI * s / (N-1));
}

double fftSmooth(double s, double ls) {
	constexpr double k = 0.8; //Smoothing Constant
	return k * ls + (1.0 - k) * s;
}

double fftClamp255(double dB) {
	constexpr double dB_min = -100.0;
	const double dB_max = -30.0;
	dB = std::clamp(dB, dB_min, dB_max);
	return std::clamp(255.0 / (dB_max - dB_min) * (dB - dB_min), 0.0, 255.0);
}

FFT::FFT(size_t nSamples, unsigned int planFlags) : nSamples(nSamples) {
	this->in = (double*)fftw_malloc(sizeof(double) * nSamples);
	this->out = (double*)fftw_malloc(sizeof(double) * nSamples);
	this->fOut = new float[nSamples];
	this->inConv = new float[nSamples];
	this->plan = fftw_plan_r2r_1d(nSamples, this->in, this->out, FFTW_R2HC, planFlags);
};

FFT::~FFT() {
	fftw_destroy_plan(this->plan);
	fftw_free(in); fftw_free(out);
	
	delete[] fOut;
	delete[] inConv;
};

void FFT::Execute(unsigned int flags) {
	// Input
	if (flags & FFT_BM_WINDOW) {
		constexpr double bmThreshold = 2048;
		for (size_t i = 0; i < this->nSamples; i++) {
			double bmWindow = fftWindow(this->in[i] * bmThreshold, this->nSamples);
			//fprintf(stdout,"Before Window: %.2lf - Window: %.2lf\n", in[i], bmWindow);
			this->in[i] *= bmWindow;
			//printf("After Window: %.2lf\n", in[i]);
		}
	}
	
	fftw_execute(this->plan);

	// Output
	for (size_t i = 0; i < this->nSamples; i++) {
		double outProcess = this->out[i];
		if (flags & FFT_ABS) outProcess = abs(outProcess);
		if (flags & FFT_ABS && (flags & FFT_CONVERT_TO_DB)) outProcess /= this->nSamples;
		this->out[i] = outProcess;
		if (flags & FFT_SMOOTH) outProcess = fftSmooth( outProcess, (i == 0 ? 0 : out[i-1]) );
		if (flags & FFT_CONVERT_TO_DB) outProcess = fftClamp255( 20*log10(outProcess) )/255;

		this->fOut[i] = (float)outProcess;
		this->inConv[i] = (float)outProcess;
	}

	if (flags & FFT_CONVOLVE) {
		double cKernel[] = { 1, 1, 2, 1, 1 };
		fftConvolution<float>(this->inConv, this->nSamples, cKernel, sizeof(cKernel) / sizeof(double), this->fOut);
	}
}