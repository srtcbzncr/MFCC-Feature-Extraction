#include "MFCC.h"
#include <iostream>
#include <stdio.h>
#include <windows.h>
#include <stdint.h>
#include <fstream>
#include <math.h>
#include <complex>
#include <cmath>

#define PI 3.1415926535897

using namespace std;

struct RIFF{
    TCHAR id[4];
    DWORD chunkSize;
    TCHAR format[4];
};

struct FMT{
    TCHAR id[4];
    DWORD chunkSize;
    WORD audioFormat;
    WORD channels;
    DWORD sampleRate;
    DWORD byteRate;
    WORD blockAlign;
    WORD bitsPerSample;
};

struct DATA{
    TCHAR id[4];
    DWORD chunkSize;
};

MFCC::MFCC(){
}

void MFCC::readFile(const char* fileName){
    cout << "Reading File is started" << endl;
    FILE *fin = fopen(fileName, "rb");
	if (fin == NULL) {
		printf("Dosya bos\n");
	}
	else {
		RIFF riff;
		FMT fmt;
		DATA data;
		fread(&riff, sizeof(riff), 1, fin);
		fread(&fmt, sizeof(fmt), 1, fin);
		fread(&data, sizeof(data), 1, fin);
		int sample_size = fmt.bitsPerSample / 8;
		int count_sample = data.chunkSize / sample_size;
		cout << "Chunk ID ->" << riff.id << endl;
		cout << "Chunk Size ->" << riff.chunkSize << endl;
		cout << "Format ->" << riff.format << endl;
		cout << "FMT ID ->" << fmt.id << endl;
		cout << "FMT Size ->" << fmt.chunkSize << endl;
		cout << "Audio Format ->" << fmt.audioFormat << endl;
		cout << "Number Of Channels ->" << fmt.channels << endl;
		cout << "Sample Rate ->" << fmt.sampleRate << endl;
		cout << "Byte Rate ->" << fmt.byteRate << endl;
		cout << "Block Align ->" << fmt.blockAlign << endl;
		cout << "Bits Per Sample ->" << fmt.bitsPerSample << endl;
		cout << "Data ID ->" << data.id << endl;
		cout << "Data Size ->" << data.chunkSize << endl;
		cout << "Sample Size ->" << sample_size << endl;
		cout << "Count Sample ->" << count_sample << endl;
		int16_t *values = new int16_t[count_sample];
		ofstream file;
        file.open("samples.txt");
		for (int i = 0; i < count_sample; i++) {
			fread(&values[i], sample_size, 1, fin);
            file << i << "->" << values[i] << endl;
		}
		file.close();
		fclose(fin);
		cout << "Reading file is successful" << endl;
        MFCC::normalization(values, count_sample, fmt.sampleRate);
	}
}

void MFCC::normalization(int16_t* samples, int sampleCount, DWORD sampleRate){
    cout << "Normalization is started" << endl;
    ofstream file;
    file.open("SamplesAfterNormalization.txt");
    double* d_samples = new double[sampleCount];
    int16_t max_sample = 0;
    for(int i=0;i<sampleCount;i++){
        if(samples[i] > max_sample){
            max_sample = samples[i];
        }
    }
    cout << "Max Sample is " << max_sample << endl;
    for(int j=0;j<sampleCount;j++){
        d_samples[j] = (double)samples[j]/max_sample;
        file << j << " -> " << d_samples[j] << endl;
    }
    file.close();
    cout << "Normalization is successful" << endl;
    MFCC::preEmphasis(d_samples, sampleCount, sampleRate, 0.95);
}

void MFCC::preEmphasis(double* samples, int sampleCount, DWORD sampleRate, double alpha){
    cout << "Pre Emphasis is started" << endl;
    double *result = new double[sampleCount];
    ofstream file;
    file.open("samplesAfterPreEmphasis.txt");
    for(int i=0;i<sampleCount;i++){
        if(i>0){
            result[i] = samples[i]-(alpha*samples[i-1]);
        }
        else{
            result[i] = samples[i]-(alpha*samples[i]);
        }
        file << i << "->" << result[i] << endl;
    }
    file.close();
    cout << "Pre Emphasis is successful" << endl;
    MFCC::framing(result, sampleCount, sampleRate);
}

void MFCC::framing(double* samples, int sampleCount, DWORD sampleRate){
    cout << "Framing is started" << endl;
    double frameDuration = 0.032;
    double offset = 0.016;
    int frameSize = sampleRate * frameDuration;
    int offsetSize = sampleRate * offset;
    int frameCount = sampleCount / offsetSize;
    int lastFrame = sampleCount % frameSize;
    if(lastFrame > 0){
        frameCount = frameCount + 1;
    }
    cout << "Frame Size ->" << frameSize << endl;
    cout << "Offset Size ->" << offsetSize << endl;
    cout << "Frame Count ->" << frameCount << endl;
    cout << "Last Frame Size ->" << lastFrame << endl;
    double **frames = new double*[frameCount];
    int start = 0;
    ofstream file;
    file.open("samplesAfterFraming.txt");
    for(int i=0;i<frameCount;i++){
        frames[i] = new double[frameSize];
        for(int j=0;j<frameSize;j++){
            frames[i][j] = samples[(i*offsetSize)+j];
            file << "Frame Number=" << i << " Sample Number=" << j << " -> " << frames[i][j] << endl;
        }
        start = start + offsetSize;
    }
    file.close();
    cout << "Framing is successful" << endl;
    MFCC::windowing(frames, frameSize, frameCount);
}

void MFCC::windowing(double** frames, int frameSize, int frameCount){
    cout << "Windowing is started" << endl;
    double *window = new double[frameSize];
    ofstream file, file2;
    file.open("windowSamples.txt");
    file2.open("samplesAfterWindowing.txt");
    for(int i=0;i<frameSize;i++){
        window[i] = 0.54 - (0.46*cos((2*PI*i)/(frameSize-1)));
        file << i << "->" << window[i] << endl;
    }
    for(int i=0;i<frameCount;i++){
        for(int j=0;j<frameSize;j++){
            frames[i][j] = frames[i][j] * window[j];
            file2 << "Frame Number=" << i << " Sample Number=" << j << " -> " << frames[i][j] << endl;
        }
    }
    file.close();
    file2.close();
    cout << "Windowing is successful" << endl;
    MFCC::FFT(frames, frameSize, frameCount);
}

void MFCC::FFT(double** frames, int frameSize, int frameCount){
    cout << "FFT is started" << endl;
    complex<double>** spectograms = new complex<double>*[frameCount];
    ofstream file;
    file.open("FFT.txt");
    for(int i=0;i<frameCount;i++){
        complex<double>* complexFrame = MFCC::frameToComplex(frames[i], frameSize);
        complex<double>* spectogram = MFCC::CooleyTukey(complexFrame, frameSize);
        spectograms[i] = spectogram;
    }
    for(int i=0;i<frameCount;i++){
        for(int j=0;j<frameSize;j++){
            file << "Frame Count -> " << i << " x[" << j <<"] -> " << spectograms[i][j] <<endl;
        }
    }
    file.close();
    cout << "FFT is successful" << endl;
    MFCC::absoulueValuesOfComplexFrequencies(spectograms, frameSize, frameCount);
}

complex<double>* MFCC::frameToComplex(double* frame, int frameSize){
    complex<double>* complexFrame = new complex<double>[frameSize];
    for(int i=0;i<frameSize;i++){
        complexFrame[i] = complex<double>(frame[i], 0);
    }
    return complexFrame;
}

complex<double>* MFCC::CooleyTukey(complex<double>* frame, int frameSize){
    if(frameSize == 1){
        return frame;
    }
    complex<double>* even = new complex<double>[frameSize/2];
    complex<double>* odd = new complex<double>[frameSize/2];
    for(int i=0;i<frameSize/2;i++){
        even[i] = frame[2*i];
        odd[i] = frame[2*i+1];
    }
    complex<double>* Feven = MFCC::CooleyTukey(even, frameSize/2);
    complex<double>* Fodd = MFCC::CooleyTukey(odd, frameSize/2);

    complex<double>* Fbins = new complex<double>[frameSize];
    for(int k=0;k<frameSize/2;k++){
        complex<double> complexExp = polar(1.0, -2*PI*k/frameSize)*Fodd[k];
        Fbins[k] = Feven[k]+complexExp;
        Fbins[k+frameSize/2] = Feven[k]-complexExp;
    }
    return Fbins;
}

void MFCC::absoulueValuesOfComplexFrequencies(complex<double>** frequencies, int frameSize, int frameCount){
    cout << "Complex Frequencies are converting to Absolute Frequencies" << endl;
    ofstream file;
    file.open("AbsoluteOfFFT.txt");
    double** absoluteFrequencies = new double*[frameCount];
    for(int i=0;i<frameCount;i++){
        double* absoluteFrequency = new double[frameSize/2+1];
        for(int j=0;j<=frameSize/2;j++){
            absoluteFrequency[j] = sqrt(pow(frequencies[i][j].real(),2)+pow(frequencies[i][j].imag(),2));
            file << "Frame Count -> " << i << " x[" << j <<"] -> " << absoluteFrequency[j] << endl;
        }
        absoluteFrequencies[i] = absoluteFrequency;
    }
    file.close();
    cout << "Complex Frequencies are converted to Absolute Frequencies" << endl;
    MFCC::powerSpectrum(absoluteFrequencies, frameSize, frameCount);
}

void MFCC::powerSpectrum(double** frequencies, int frameSize, int frameCount){
    cout << "Power Spectrum is calculating" << endl;
    ofstream file;
    file.open("PowerSpectrum.txt");
    double** powerSpectrums = new double*[frameCount];
    for(int i=0;i<frameCount;i++){
        double* powerSpectrum = new double[frameSize/2+1];
        for(int j=0;j<=frameSize/2;j++){
            powerSpectrum[j] = pow(frequencies[i][j],2)/frameSize;
            file << "Frame Count -> " << i << " P[" << j <<"] -> " << powerSpectrum[j] << endl;
        }
        powerSpectrums[i] = powerSpectrum;
    }
    file.close();
    cout << "Power Spectrum is calculated" << endl;
    MFCC::calculateMFCC(powerSpectrums, frameSize, frameCount, 26);
}

void MFCC::calculateMFCC(double** powerSpectrums, int frameSize, int frameCount, int filterCount){
    cout << "Mel Frequency Cepstral Cofficients are calculating" << endl;
    ofstream file, file2;
    file.open("coefficients.txt");
    file2.open("logCoefficients.txt");
    double** coefficients = new double*[frameCount];
    double** filterBanks = MFCC::melFilterBank(300.0, 8000.0, frameSize, filterCount);
    for(int i=0;i<frameCount;i++){
        double* powerSpectrum = powerSpectrums[i];
        double* coefficient = new double[filterCount];
        for(int j=0;j<filterCount;j++){
            double sum = 0.0;
            for(int k=0;k<=frameSize/2;k++){
                sum = sum + powerSpectrum[k]*filterBanks[j][k];
            }
            coefficient[j] = sum;
        }
        coefficients[i] = coefficient;
    }
    double** logCoefficients = MFCC::getLogOfSum(coefficients, frameCount, filterCount);
    for(int i=0;i<frameCount;i++){
        for(int j=0;j<filterCount;j++){
            file << "coefficients[" << i << "][" << j << "] -> " << coefficients[i][j] << endl;
            file2 << "logCoefficients[" << i << "][" << j << "] -> " << logCoefficients[i][j] << endl;
        }
    }
    file.close();
    file2.close();
    cout << "Mel Frequency Cepstral Coefficients are calculated" << endl;
    MFCC::DCT(coefficients, frameCount, filterCount);
}

double** MFCC::getLogOfSum(double** coefficients, int frameCount, int filterCount){
    double** logCoefficients = new double*[frameCount];
    for(int i=0;i<frameCount;i++){
        double* logCoefficient = new double[filterCount];
        for(int j=0;j<filterCount;j++){
            logCoefficient[j] = log(coefficients[i][j]);
        }
        logCoefficients[i] = logCoefficient;
    }
    return logCoefficients;
}

double** MFCC::melFilterBank(double lowerFrequency, double upperFrequency, int frameSize, int filterCount){
    cout << "Mel Filterbank is calculating" << endl;
    double lowerMel = MFCC::frequencyToMel(lowerFrequency);
    double upperMel = MFCC::frequencyToMel(upperFrequency);
    double* mFilters = new double[filterCount+2];
    double* fFilters = new double[filterCount+2];
    int* fbinFilters = new int[filterCount+2];
    double gap = (upperMel - lowerMel) / (filterCount+1);
    mFilters[0] = lowerMel;
    fFilters[0] = melToFrequency(lowerMel);
    mFilters[filterCount+1] = upperMel;
    fFilters[filterCount+1] = melToFrequency(upperMel);
    fbinFilters[0] = floor((frameSize+1)*fFilters[0]/16000);
    fbinFilters[filterCount+1] = floor((frameSize+1)*fFilters[filterCount+1]/16000);
    for(int i=1;i<filterCount+1;i++){
        mFilters[i] = mFilters[i-1]+gap;
        fFilters[i] = melToFrequency(mFilters[i]);
        fbinFilters[i] = floor((frameSize+1)*fFilters[i]/16000);
     }
     double** filterBanks = new double*[filterCount];
     for(int m=0;m<filterCount;m++){
            double* filterBank = new double[(frameSize/2)+1];
        for(int k=0;k<=frameSize/2;k++){
            if(k < fbinFilters[m]){
                filterBank[k] = 0.0;
            }
            else if(k < fbinFilters[m+1]){
                double value1 = k-fbinFilters[m];
                double value2 = fbinFilters[m+1]-fbinFilters[m];
                filterBank[k] = value1/value2;
            }
            else if(k < fbinFilters[m+2]){
                double value1 = fbinFilters[m+2]-k;
                double value2 = fbinFilters[m+2]-fbinFilters[m+1];
                filterBank[k] = value1/value2;
            }
            else{
                filterBank[k] = 0.0;
            }
            filterBanks[m] = filterBank;
        }
     }
     cout << "Mel Filterbank is calculated" << endl;
     return filterBanks;
}

double MFCC::frequencyToMel(double frequency){
    return 1125*log(1+frequency/700);
}

double MFCC::melToFrequency(double mel){
    complex<double> cmplx = exp(mel/1125);
    return 700*(cmplx.real()-1);
}

void MFCC::DCT(double** coefficients, int frameCount, int filterCount){
    cout << "DCT is calculating" << endl;
    ofstream file;
    file.open("cepstrumsAfterDCT.txt");
    double e1 = sqrt((2/(double)filterCount));
    double** results = new double*[frameCount];
    for(int i=0;i<frameCount;i++){
        double* result = new double[14];
        for(int u=0;u<14;u++){
            double sum = 0.0;
            double a;
            if(u==0){
                a = 1/(double)sqrt(2);
            }
            else{
                a = 1.0;
            }
            for(int n=0;n<filterCount;n++){
                double c = (PI*u*(2*n+1))/(2*(double)filterCount);
                sum = sum + cos(c)*coefficients[i][n];
            }
            result[u] = sum*e1*a;
            file << "Frame[" << i << "] Coefficient[" << u << "] -> " << result[u] << endl;
        }
        results[i] = result;
    }
    file.close();
    cout << "DCT is calculated" << endl;
}

MFCC::~MFCC(){
}
