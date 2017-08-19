#!/usr/bin/env python

from python_speech_features import bfcc as xfcc
from python_speech_features import delta
from python_speech_features import logfbank, hz2bark, bark2hz
import scipy.io.wavfile as wav

(rate, sig) = wav.read("english.wav")
xfcc_feat = xfcc(sig, rate)
d_xfcc_feat = delta(xfcc_feat, 2)
# fbank_feat = logfbank(sig, rate, scaling=(hz2bark, bark2hz))
#
# print(fbank_feat[1:3, :])


print(xfcc_feat)