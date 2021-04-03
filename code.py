import os
from os.path import isdir, join
from scipy.io import wavfile
from subprocess import check_output
from pathlib import Path
import pandas as pd


# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

%matplotlib inline

!pip install pyunpack
!pip install patool

from pyunpack import Archive
import shutil
if not os.path.exists('/kaggle/working/train/'):
    os.makedirs('/kaggle/working/train/')
Archive('/kaggle/input/train.7z').extractall('/kaggle/working/train/')
for dirname, _, filenames in os.walk('/kaggle/working/train/'):
    for filename in filenames:
        os.path.join(dirname, filename)
#       print(os.path.join(dirname, filename))

shutil.make_archive('train/', 'zip', 'train')

# deleting unwanted extracted files to avoid memory overflow while commiting.
!rm -rf kaggle/working/train/*
# Loading the training Input file.
train_audio_path = "/kaggle/working/train/train/audio"
# Checking to validate the presence of file.

print(check_output(["ls", "/kaggle/working/train/train/audio"]).decode("utf8"))
print(os.listdir("/kaggle/working/train/train/audio/yes"))

# As an example using this input file here...
filename = '/yes/00f0204f_nohash_0.wav'

dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
dirs.sort()
print('Number of labels: ' + str(len(dirs)))

samples, sample_rate = librosa.load(str(train_audio_path)+filename)

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)
  
freqs, times, spectrogram = log_specgram(samples, sample_rate)

fig = plt.figure(figsize=(14, 8))
plot1 = fig.add_subplot(211)
plot1.set_title('Raw wave of ' + filename)
plot1.set_ylabel('Amplitude')
plot1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

plot2 = fig.add_subplot(212)
plot2.imshow(spectrogram.T, aspect='auto', origin='lower', 
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
plot2.set_yticks(freqs[::16])
plot2.set_xticks(times[::16])
plot2.set_title('Spectrogram of ' + filename)
plot2.set_ylabel('Frequency in Hz')
plot2.set_xlabel('Seconds')

mean = np.mean(spectrogram, axis=0)
std = np.std(spectrogram, axis=0)
spectrogram = (spectrogram - mean) / std

# From this tutorial
# https://github.com/librosa/librosa/blob/master/examples/LibROSA%20demo.ipynb
S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)

# Convert to log scale (dB). Using the peak power (max) as reference.
log_S = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
plt.title('Mel power spectrogram ')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()


mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

# Let's pad on the first and second deltas while we're at it
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

plt.figure(figsize=(12, 4))
librosa.display.specshow(delta2_mfcc)
plt.ylabel('MFCC coeffs')
plt.xlabel('Time')
plt.title('MFCC')
plt.colorbar()
plt.tight_layout()

samples_cut = samples[4000:13000]
ipd.Audio(samples_cut, rate=sample_rate)


freqs, times, spectrogram_cut = log_specgram(samples_cut, sample_rate)

fig = plt.figure(figsize=(14, 8))
plot1 = fig.add_subplot(211)
plot1.set_title('Raw wave of ' + filename)
plot1.set_ylabel('Amplitude')
plot1.plot(samples_cut)

plot2 = fig.add_subplot(212)
plot2.set_title('Spectrogram of ' + filename)
plot2.set_ylabel('Frequencies * 0.1')
plot2.set_xlabel('Samples')
plot2.imshow(spectrogram_cut.T, aspect='auto', origin='lower', 
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
plot2.set_yticks(freqs[::16])
plot2.set_xticks(times[::16])
plot2.text(0.06, 1000, 'Y', fontsize=16)
plot2.text(0.17, 1000, 'E', fontsize=16)
plot2.text(0.36, 1000, 'S', fontsize=16)

xcoords = [0.025, 0.11, 0.23, 0.49]
for xc in xcoords:
    plot1.axvline(x=xc*16000, c='r')
    plot2.axvline(x=xc, c='r')


!pip install webrtcvad
import webrtcvad

sample_rate, samples = wavfile.read(str(train_audio_path) + filename)

vad = webrtcvad.Vad()

# set aggressiveness from 0 to 3
vad.set_mode(3)

import struct
raw_samples = struct.pack("%dh" % len(samples), *samples)

window_duration = 0.03 # duration in seconds

samples_per_window = int(window_duration * sample_rate + 0.5)

bytes_per_sample = 2

segments = []

for start in np.arange(0, len(samples), samples_per_window):
    stop = min(start + samples_per_window, len(samples))
    
    is_speech = vad.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], 
                              sample_rate = sample_rate)

    segments.append(dict(
       start = start,
       stop = stop,
       is_speech = is_speech))
    
    
plt.figure(figsize = (10,7))
plt.plot(samples)

ymax = max(samples)


# plot segment identifed as speech
for segment in segments:
    if segment['is_speech']:
        plt.plot([ segment['start'], segment['stop'] - 1], [ymax * 1.1, ymax * 1.1], color = 'purple')

plt.xlabel('sample')
plt.grid()


speech_samples = np.concatenate([ samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])

import IPython.display as ipd
ipd.Audio(speech_samples, rate=sample_rate)

def violinplot_frequency(dirs, freq_ind):
    """ Plot violinplots for given words (waves in dirs) and frequency freq_ind
    from all frequencies freqs."""

    spec_all = []  # Contain spectrograms
    ind = 0
    # taking first 8 words only to keep the plots clean and unclumsy.
    for direct in dirs[:8]:
        spec_all.append([])

        waves = [f for f in os.listdir(join(train_audio_path, direct)) if
                 f.endswith('.wav')]
        for wav in waves[:100]:
            sample_rate, samples = wavfile.read(
                train_audio_path + '/' + direct + '/' + wav)
            freqs, times, spec = log_specgram(samples, sample_rate)
            spec_all[ind].extend(spec[:, freq_ind])
        ind += 1

    # Different lengths = different num of frames. Make number equal
    minimum = min([len(spec) for spec in spec_all])
    spec_all = np.array([spec[:minimum] for spec in spec_all])

    plt.figure(figsize=(13,7))
    plt.title('Frequency ' + str(freqs[freq_ind]) + ' Hz')
    plt.ylabel('Amount of frequency in a word')
    plt.xlabel('Words')
    sns.violinplot(data=pd.DataFrame(spec_all.T, columns=dirs[:8]))
    plt.show()
    violinplot_frequency(dirs, 20)
    violinplot_frequency(dirs, 50)
    violinplot_frequency(dirs, 1200)
