"""
comments from notebook:
# Sampling frequency is 160 Hz
# With 32 samples the frequency resolution after FFT is 160 / 32
"""
import os
import pickle

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import librosa
from librosa import feature
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

n_fft = 32
win_length = None
hop_length = 4
mel_spectrogram = T.MelSpectrogram(
    n_mels=10,
    sample_rate=160,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    normalized=True,
    f_min=0,
    f_max=80
)



def compute_spectrogram(signal, title):
    """
    outputs:
        freq_signal = list of 8 Tensors (one arm at a time)
    """
    if True:
        # how many channels do we have (should always be 8)
        n_channels = signal.shape[1]
        # list of n_channels spectrograms (2D np.array)
        freq_signal = [mel_spectrogram(torch.Tensor(signal[:, i])) for i in range(n_channels)]

        plot_spectrogram(freq_signal, title=title)

    return freq_signal



def spectrogram_main(input_file, output_file, isSubaction):
    with open(input_file, 'rb') as preprocessed_emg:
        pkl_dict = dict()
        for pre_index, pre_dict in pickle.load(preprocessed_emg).items():
            signal_left = pre_dict['preprocessed-left-data']
            signal_right = pre_dict['preprocessed-right-data']
            label = pre_dict['label']
            left_right_append = list()


            # # LEFT
            # needed to compute spectrograms
            signal_left = signal_left.astype(float)
            # compute and plot spectrograms
            spect_left = compute_spectrogram(signal_left, label)

            # # RIGHT
            # needed to compute spectrograms
            signal_right = signal_right.astype(float)
            # compute and plot spectrograms
            spect_right = compute_spectrogram(signal_right, label)

            # append left and right spectrogram to final list
            left_right_append.extend(spect_left)
            left_right_append.extend(spect_right)

            # save new dict
            if isSubaction:
                pkl_dict[pre_index] = {
                    "index": pre_index,
                    "label": pre_dict["label"],
                    "spectrograms-list": left_right_append,
                    "frames-sub_indexes-array": pre_dict["frames-sub_indexes-array"]

                }
            else:
                pkl_dict[pre_index] = {
                    "index": pre_index,
                    "label": pre_dict["label"],
                    "spectrograms-list": left_right_append,
                }

        # SAVE dictionary in pickle file
        pickle.dump(pkl_dict, open(str(output_file + ".pkl"), "wb"))


if __name__ == "__main__":
    # NO SUBACTION -> we are running directly from spectrogram
    input_file = "./preprocessed_EMG/S04_EMG.pkl"
    out_dir = "spectrogram_EMG"
    output_path = os.path.join(out_dir, "S04_spectrogram")
    spectrogram_main(input_file, output_path, isSubaction=False)

