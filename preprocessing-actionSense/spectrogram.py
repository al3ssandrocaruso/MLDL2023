"""
comments from notebook:
# Sampling frequency is 160 Hz
# With 32 samples the frequency resolution after FFT is 160 / 32
"""

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(len(specgram), 1, figsize=(16, 8))

    axs[0].set_title(title or "Spectrogram (db)")

    for i, spec in enumerate(specgram):
        im = axs[i].imshow(librosa.power_to_db(specgram[i]), origin="lower", aspect="auto")
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    axs[i].set_xlabel("Frame number")
    axs[i].get_xaxis().set_visible(True)
    plt.show(block=False)

"""
If you want to use TORCHAUDIO uncomment this and comment the function below
"""
# n_fft = 32
# win_length = None
# hop_length = 4
# spectrogram = T.Spectrogram(
#     n_fft=n_fft,
#     win_length=win_length,
#     hop_length=hop_length,
#     center=True,
#     pad_mode="reflect",
#     power=2.0,
#     normalized=True
# )

# def spectrogram(channel):
#     n_fft = 32
#     win_length = None
#     hop_length = 4
#     return librosa.stft(
#         y=channel,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         win_length=win_length,
#         center=True,
#         pad_mode="reflect"
#     )

def spectrogram(channel):
    sr = 160
    end_freq = 5
    start_freq = 0
    # Compute the corresponding number of FFT bins
    n_fft = int(sr / end_freq) * 2  # Choose a suitable value based on the highest frequency of interest
    # Compute the corresponding window length
    # win_length = int(sr / start_freq) * 2  # Choose a suitable value based on the lowest frequency of interest

    return librosa.stft(
        y=channel,
        n_fft=n_fft,
        # win_length=win_length
    )


def compute_spectrogram(signal, title):
    # if one single channel is passed (activation)
    if len(signal.shape) == 1:
        freq_signal = spectrogram(signal)

        # Plot the spectrogram
        plt.figure(figsize=(16, 8))
        librosa.display.specshow(freq_signal, sr=160, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()

    else:
        # how many channels do we have (should always be 8)
        n_channels = signal.shape[1]
        # list of n_channels spectrograms (2D np.array)
        freq_signal = [spectrogram(signal[:, i]) for i in range(n_channels)]
        print("Freq size", freq_signal[0].shape)
        plot_spectrogram(freq_signal, title=title)

    return freq_signal



def main():
    # Replace with your path to one of the subjects from Action-Net
    emg_annotations = pd.read_pickle("./preprocessed_EMG/S04_EMG.pkl")

    # extract signal from pickle file
    sample_no = 1
    signal = emg_annotations[1]['preprocessed_activation_left']
    # signal = emg_annotations[1]['myo-left-data']
    title = emg_annotations[1]['label']

    # needed to compute spectrograms
    signal = signal.astype(float)

    # compute and plot spectrograms, freq_spect is a list of 1 (only activation) or 8 (all channels) spectrograms
    freq_signal = compute_spectrogram(signal, title)
    for i in range(len(freq_signal)):
        freq_signal[i] = librosa.stft
    print(freq_signal)


if __name__ == "__main__":
    main()

