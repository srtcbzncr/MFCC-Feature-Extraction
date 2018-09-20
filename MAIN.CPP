#include <iostream>
#include "MFCC.h"
#include <stdint.h>
#include <stdio.h>

using namespace std;

int main()
{
    MFCC mfcc;
    mfcc.readFile("arctic_a0001.wav");
    cout << "Cikis icin e'ye basiniz" << endl;
    while(true){
        char key;
        cin >> key;
        if(key == 'e'){
            break;
        }
    }
    remove("samples.txt");
    remove("SamplesAfterNormalization.txt");
    remove("samplesAfterPreEmphasis.txt");
    remove("samplesAfterFraming.txt");
    remove("samplesAfterWindowing.txt");
    remove("windowSamples.txt");
    remove("FFT.txt");
    remove("AbsoluteOfFFT.txt");
    remove("PowerSpectrum.txt");
    remove("coefficients.txt");
    remove("logCoefficients.txt");
    remove("cepstrumsAfterDCT.txt");
    return 0;
}
