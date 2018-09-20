#ifndef MFCC_H_INCLUDED
#define MFCC_H_INCLUDED
#include <stdint.h>
#include <windows.h>
#include <complex>

using namespace std;

class MFCC{
public:
    MFCC();
    void readFile(const char* fileName);
    void normalization(int16_t* samples, int sampleCount, DWORD sampleRate);
    void preEmphasis(double* samples, int sampleCount, DWORD sampleRate, double alpha);
	void framing(double* samples, int sampleCount, DWORD sampleRate);
	void windowing(double** frames, int frameSize, int frameCount);
	void FFT(double** frames, int frameSize, int frameCount);
	complex<double>* frameToComplex(double* frame, int frameSize);
	complex<double>* CooleyTukey(complex<double>* frame, int frameSize);
	void absoulueValuesOfComplexFrequencies(complex<double>** frequencies, int frameSize, int frameCount);
	void powerSpectrum(double** frequencies, int frameSize, int frameCount);
	void calculateMFCC(double** powerSpectrums, int frameSize, int frameCount, int filterCount);
	double** melFilterBank(double lowerFrequency, double upperFrequency, int frameSize, int filterCount);
	double frequencyToMel(double frequency);
	double melToFrequency(double mel);
	double** getLogOfSum(double** coefficients, int frameCount, int filterCount);
	void DCT(double** coefficients, int frameCount, int filterCount);
	void vectorQuantization(double** coefficients, int frameCount, int filterCount, int clusterSize, double e);
	double* createCodeBook(double** coefficients, int frameCount, int filterCount);
	double** expandCodeBook(double** coefficients, double** codeBook, int frameCount, int filterCount, int clusterSize, double e, int iteration);
    double* nearestSearch(double** coefficients, double** codeBook, int frameCount, int filterCount, int clusterSize);
	~MFCC();
};
#endif // MFCC_H_INCLUDED
