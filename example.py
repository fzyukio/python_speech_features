#!/usr/bin/env python

from python_speech_features import xfcc
from python_speech_features import delta
from python_speech_features import logfbank, hz2bark, bark2hz
import scipy.io.wavfile as wav

(rate, sig) = wav.read("english.wav")
mfcc_feat = xfcc('mfcc', signal=sig, samplerate=rate, cepsfunc='pha')

# d_mfcc_feat = delta(mfcc_feat, 2)
# fbank_feat = logfbank(sig, rate, scaling=(hz2bark, bark2hz))
#
# print(fbank_feat[1:3, :])

print('MFCC features: ')
print(mfcc_feat)

