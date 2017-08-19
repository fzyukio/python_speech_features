#!/usr/bin/env python

from python_speech_features import bfcc, mfcc, gfcc
from python_speech_features import delta
from python_speech_features import logfbank, hz2bark, bark2hz
import scipy.io.wavfile as wav

(rate, sig) = wav.read("english.wav")
mfcc_feat = mfcc(signal=sig, samplerate=rate)
bfcc_feat = bfcc(signal=sig, samplerate=rate)

lowhear = 500
hihear = 12000

gfcc_feat = gfcc(signal=sig, samplerate=rate, lowhear=lowhear, hihear=hihear)

# d_mfcc_feat = delta(mfcc_feat, 2)
# fbank_feat = logfbank(sig, rate, scaling=(hz2bark, bark2hz))
#
# print(fbank_feat[1:3, :])

print('MFCC features: ')
print(mfcc_feat)

print('BFCC features: ')
print(bfcc_feat)

print('GFCC features with lowhear={}Hz, hihear={}Hz:'.format(lowhear, hihear))
print(gfcc_feat)
