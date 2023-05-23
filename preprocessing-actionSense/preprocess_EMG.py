"""
preprocess data from input pickle.
saves to output pickle NEW DICTIONARY
    "index": emg_index,
    "label": emg_dict["label"],
    "preprocessed-left-data": emg_left_preproc,
    "preprocessed-right-data": emg_right_preproc,
CHANGES: preprocess-left/right-data instead of raw data
"""
import pickle
import numpy as np
import os
import scipy.signal
import spectrogram

# specify directory of the file to preprocess
directory = "./parsed_EMG"

# specify frequency of myo sensor
FS = 160

def _rms(x, win):
    output = np.zeros((x.shape))
    npad = np.floor(win / 2).astype(int)
    win = int(win)
    x_ = np.pad(x, ((npad, npad), (0, 0)), 'symmetric')
    for i in range(output.shape[0]):
        output[i, :] = np.sqrt(np.sum(x_[i:i + win, :] ** 2, axis=0) / win)
    return output

def rms(x, fs=FS):
    win = 0.2 * fs
    return _rms(x, win)

def lpf(x, f=1., fs=FS):
    f = f / (FS / 2)
    x = np.abs(x)
    b, a = scipy.signal.butter(4, f, 'low')
    output = scipy.signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
    return output


def preprocess_data(emg, plot_label):
    # plot_emg(emg[:, 0], plot_label + " before rms")
    emg = rms(emg)
    # plot_emg(emg[:, 0], plot_label + " after rms")

    # min-max [0, 1] normalization
    emg_min = emg.min()
    emg_max = emg.max()
    emg = (emg - emg_min) / (emg_max - emg_min)
    # plot_emg(emg[:, 0], plot_label + " after min-max")

    # low pass filter
    emg = lpf(emg)
    # plot_emg(emg[:, 0], plot_label + " after lpf")

    # # Plot of emg signal after preprocessing
    # plot_emg(emg, plot_label)

    return emg


def main():
    for parsed_emg_filename in [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]:
        parsed_emg_filepath = os.path.join(directory, parsed_emg_filename)
        with open(parsed_emg_filepath, 'rb') as parsed_emg:
            pkl_dict = dict()
            for emg_index, emg_dict in pickle.load(parsed_emg).items():
                emg_left_data = emg_dict['myo-left-data']
                emg_right_data = emg_dict['myo-right-data']

                # # LEFT
                emg_left_preproc = preprocess_data(emg_left_data, "left")

                # # RIGHT
                emg_right_preproc = preprocess_data(emg_right_data, "right")

                # save new dict
                pkl_dict[emg_index] = {
                    "index": emg_index,
                    "label": emg_dict["label"],
                    "preprocessed-left-data": emg_left_preproc,
                    "myo-left-timestamps": emg_dict["myo-left-timestamps"],
                    "preprocessed-right-data": emg_right_preproc,
                    "myo-right-timestamps": emg_dict["myo-right-timestamps"],
                    "start_time": emg_dict["start_time"],
                    "end_time": emg_dict["end_time"],
                    "duration": emg_dict["duration"]
                }

            # SAVE dictionary in pickle file
            out_dir = "preprocessed_EMG"
            output_path = os.path.join(out_dir, parsed_emg_filename.split('_')[0] + "_EMG")
            pickle.dump(pkl_dict, open(str(output_path + ".pkl"), "wb"))


if __name__ == "__main__":
    main()
