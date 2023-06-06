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

n_fft = 16
win_length = None
hop_length = 2
mel_spectrogram_CLIP = T.MelSpectrogram(
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



def compute_spectrogram_CLIP(signal, title):
    """
    outputs:
        freq_signal = list of 8 Tensors (one arm at a time)
    """
    # how many channels do we have (should always be 8)
    n_channels = signal.shape[1]
    # list of n_channels spectrograms (2D np.array)
    freq_signal = [mel_spectrogram_CLIP(torch.Tensor(signal[:, i])) for i in range(n_channels)]

    # plot_spectrogram(freq_signal, title=title)

    return freq_signal




def spectrogram_main_CLIP(input_file, output_file, n_clips, isSubaction):
    """
    we have 5 clips, original emg signal is (800x8) after interpolation
    each clip is (160x8)
    each spectrogram with reduced N_FFT and HOP_LENGTH is
        (10x81)
    """
    with open(input_file, 'rb') as preprocessed_emg:
        print(input_file)
        pkl_dict = dict()
        for pre_index, pre_dict in pickle.load(preprocessed_emg).items():
            left_clip_list = pre_dict["preprocessed-left-clip-list"]
            right_clip_list = pre_dict["preprocessed-right-clip-list"]
            label = pre_dict['label']
            clip_spectrograms_list = list()

            for i in range(n_clips):
                left_right_append = list()

                # # LEFT
                # extract clip
                left_clip = left_clip_list[i]
                # needed to compute spectrograms
                left_clip = left_clip.astype(float)
                # compute and plot spectrograms
                spect_left = compute_spectrogram_CLIP(left_clip, label)

                # print("clip spectrogram", spect_left[0].shape)

                # # RIGHT
                # extract clip
                right_clip = right_clip_list[i]
                # needed to compute spectrograms
                right_clip = right_clip.astype(float)
                # compute and plot spectrograms
                spect_right = compute_spectrogram_CLIP(right_clip, label)

                # print("clip spectrogram", spect_left[0].shape)

                # append left and right spectrogram to final list for this clip
                left_right_append.extend(spect_left)
                left_right_append.extend(spect_right)

                # stack here and not in colab
                left_right_append = torch.stack(left_right_append)

                # LIST OF LIST -> 5 lists, each list is a clip and has the 16 spectrograms
                clip_spectrograms_list.append(left_right_append)

            print(f"done sub_action {pre_index}")
            # save new dict
            if isSubaction:
                pkl_dict[pre_index] = {
                    "index": pre_index,
                    "label": pre_dict["label"],
                    "clip_spectrograms_list": clip_spectrograms_list,
                    # "frames-sub_indexes-array": pre_dict["frames-sub_indexes-array"]
                }
            else:
                raise ValueError
                # pkl_dict[pre_index] = {
                #     "index": pre_index,
                #     "label": pre_dict["label"],
                #     "spectrograms-list": left_right_append,
                # }
        
        # SAVE dictionary in pickle file
        pickle.dump(pkl_dict, open(str(output_file + ".pkl"), "wb"))


if __name__ == "__main__":
    raise ValueError
    # # NO SUBACTION -> we are running directly from spectrogram
    # input_file = "./preprocessed_EMG/S04_EMG.pkl"
    # out_dir = "spectrogram_EMG"
    # output_path = os.path.join(out_dir, "S04_spectrogram_CLIP")
    # n_clips = 5
    # spectrogram_main_CLIP(input_file, output_path, n_clips, isSubaction=False)

