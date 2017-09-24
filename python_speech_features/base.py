# calculate filterbank features. Provides e.g. fbank and mfcc features for use in ASR applications
# Author: James Lyons 2012
from __future__ import division
import numpy
from python_speech_features import sigproc
from scipy.fftpack import dct


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1 + hz / 700.)


def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def hz2bark(hz):
    """
    Convert hertz to Bark scale value
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :return: a value in Barks. If an array was passed in, an identical sized array is returned.
    """
    return 13 * numpy.arctan(0.00076 * hz) + 3.5 * numpy.arctan((hz/7500) ** 2)


def bark2hz(bark):
    """Convert a value in Mels to Hertz

    :param bark: a value in Barks. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 52548 / (bark ** 2 - 52.56 * bark + 690.39)


def hz2hz(hz):
    "Dummy converter"
    return hz


def greenwood(fmin, fmax, k=0.88):
    """
    Generate a pair of greenwood scale converters. This will generate hz2mel and mel2hz if
    :param fmin: lower range of hearing
    :param fmax: higher range of hearing
    :param k: balancing coefficient
    :return: a tuple of two converters
    """
    A = fmin / (1-k)
    a = numpy.log10(fmax / A + k)
    from_hz = lambda hz: 1/a * numpy.log10(k + hz / float(A))
    to_hz = lambda gw: float(A) * (10 ** (gw * a) - k)
    return from_hz, to_hz


def xfcc(scaling, signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0,
         highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=lambda x: numpy.ones((x,))):
    """Compute xFCC features from an audio signal, given a scale

    :param scaling: a pair of conversion function to convert Hz to and from another scale, such as Mel, Bark,...
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame
                        energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy
                    window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc, scaling)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy: feat[:, 0] = numpy.log(energy)  # replace first cepstral coefficient with log of frame energy
    return feat


def mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
         preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=lambda x: numpy.ones((x,))):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame
                        energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy
                    window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    return xfcc((hz2mel, mel2hz), signal, samplerate, winlen, winstep, numcep, nfilt, nfft, lowfreq, highfreq, preemph,
                ceplifter, appendEnergy, winfunc)


def lfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
         preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=lambda x: numpy.ones((x,))):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame
                        energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy
                    window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    return xfcc((hz2hz, hz2hz), signal, samplerate, winlen, winstep, numcep, nfilt, nfft, lowfreq, highfreq, preemph,
                ceplifter, appendEnergy, winfunc)


def bfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
         preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=lambda x: numpy.ones((x,))):
    """Compute Bark-scale features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame
                        energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy
                    window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    return xfcc((hz2bark, bark2hz), signal, samplerate, winlen, winstep, numcep, nfilt, nfft, lowfreq, highfreq,
                preemph, ceplifter, appendEnergy, winfunc)


def gfcc(lowhear, hihear, signal, k=0.88, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512,
         lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=lambda x: numpy.ones((x,))):
    """Compute Greenwood scale features from an audio signal, given the species' range of hearing

    :param hihear: The upper frequency of the range of hearing of this species
    :param lowhear: The lower frequency of the range of hearing of this species
    :param k: The species specific balancing parameter
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame
                        energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy
                    window functions here e.g. winfunc=numpy.hamming
    :param scaling: a pair of conversion function to convert Hz to and from another scale, such as Mel, Bark,...
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """

    scaling = greenwood(lowhear, hihear, k)
    return xfcc(scaling, signal, samplerate, winlen, winstep, numcep, nfilt, nfft, lowfreq, highfreq, preemph,
                ceplifter, appendEnergy, winfunc)


def fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
          preemph=0.97, winfunc=lambda x: numpy.ones((x,)), scaling=(hz2mel, mel2hz)):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy
                    window functions here e.g. winfunc=numpy.hamming
    :param scaling: a pair of conversion function to convert Hz to and from another scale, such as Mel, Bark,...
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1
              feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq = highfreq or samplerate / 2
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = sigproc.powspec(frames, nfft)
    energy = numpy.sum(pspec, 1)  # this stores the total energy in each frame
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)  # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq, scaling)
    feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)  # if feat is zero, we get problems with log

    return feat, energy


def logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
             preemph=0.97, scaling=(hz2mel, mel2hz)):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param scaling: a pair of conversion function to convert Hz to and from another scale, such as Mel, Bark,...
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, scaling=scaling)
    return numpy.log(feat)


def ssc(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
        preemph=0.97, winfunc=lambda x: numpy.ones((x,)), scaling=(hz2mel, mel2hz)):
    """Compute Spectral Subband Centroid features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy
                    window functions here e.g. winfunc=numpy.hamming
    :param scaling: a pair of conversion function to convert Hz to and from another scale, such as Mel, Bark,...
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    highfreq = highfreq or samplerate / 2
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = sigproc.powspec(frames, nfft)
    pspec = numpy.where(pspec == 0, numpy.finfo(float).eps, pspec)  # if things are all zeros we get problems

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq, scaling)
    feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
    R = numpy.tile(numpy.linspace(1, samplerate / 2, numpy.size(pspec, 1)), (numpy.size(pspec, 0), 1))

    return numpy.dot(pspec * R, fb.T) / feat


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None, scaling=(hz2mel, mel2hz)):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :param scaling: a pair of conversion function to convert Hz to and from another scale, such as Mel, Bark,...
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels (or barks)
    from_hz = scaling[0]
    to_hz = scaling[1]

    low_scale = from_hz(lowfreq)
    high_scale = from_hz(highfreq)
    mel_points = numpy.linspace(low_scale, high_scale, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bins = numpy.floor((nfft + 1) * to_hz(mel_points) / samplerate)

    fbanks = numpy.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bins[j]), int(bins[j + 1])):
            fbanks[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            fbanks[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    return fbanks


def lifter(cepstra, liftering_coefs=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param liftering_coefs: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if liftering_coefs > 0:
        num_frames, num_coeff = numpy.shape(cepstra)
        n = numpy.arange(num_coeff)
        lift = 1 + (liftering_coefs / 2.) * numpy.sin(numpy.pi * n / liftering_coefs)
        return lift * cepstra
    else:
        # values of liftering_coefs <= 0, do nothing
        return cepstra


def delta(feat, N):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature
                 vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta
              feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    num_frames = len(feat)
    denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((N, N), (0, 0)), mode='edge')  # padded version of feat
    for t in range(num_frames):
        delta_feat[t] = numpy.dot(numpy.arange(-N, N + 1),
                                  padded[t: t + 2 * N + 1]) / denominator  # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat
