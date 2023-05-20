"""
Structure of HDF5 file: similar to a dictionary of dictionaries of dictionaries... (key: value), same way to access
first keys: ['experiment-activities', 'experiment-calibration', 'experiment-notes', 'eye-tracking-gaze', 'eye-tracking-pupil', 'eye-tracking-time', 'eye-tracking-video-eye', 'eye-tracking-video-world', 'eye-tracking-video-worldGaze', 'myo-left', 'myo-right', 'tactile-calibration-scale', 'tactile-glove-left', 'tactile-glove-right', 'xsens-CoM', 'xsens-ergonomic-joints', 'xsens-foot-contacts', 'xsens-joints', 'xsens-segments', 'xsens-sensors', 'xsens-time']>
'myo-left', 'myo-right' keys: ['acceleration_g', 'angular_velocity_deg_s', 'battery', 'emg', 'gesture', 'orientation_quaternion', 'rssi', 'synced']
'emg' keys: ['data', 'time_s', 'time_s_original', 'time_str']
"""
import pickle
import numpy as np
import os
from scipy.signal import butter, filtfilt
import spectrogram

# specify directory of the file to preprocess
directory = "./parsed_EMG"


def plot_emg(emg, plot_label, fs=160):
    import matplotlib.pyplot as plt

    # Create a time vector for the x-axis
    n_samples = len(emg)
    t = np.arange(n_samples) / fs

    # Plot the EMG signal
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, emg, label='Raw EMG')

    ax.legend(loc='upper right')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('EMG Signal: {}'.format(plot_label))

    plt.show()


def low_pass(emg, order=2, cutoff_freq=5, fs=160):
    nyquist_freq = 0.5 * fs
    cutoff_norm = cutoff_freq / nyquist_freq

    # Define the filter coefficients
    b, a = butter(N=order, Wn=cutoff_norm, fs=fs, btype='lowpass', analog=False, output='ba')
    # Reshape
    emg = emg.reshape(-1)
    # Apply the filter
    emg = filtfilt(b, a, emg)


###############################################################################################
import scipy.signal
import numpy as np

FS = 160


# SUBSAMPLE_FACTOR = 20

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
    b, a = scipy.signal.butter(1, f, 'low')
    output = scipy.signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
    return output


###############################################################################################


def preprocess_data(emg, plot_label):
    plot_emg(emg[:, 0], plot_label + " before rms")
    emg = rms(emg)
    plot_emg(emg[:, 0], plot_label + " after rms")

    # min-max [0, 1] normalization
    emg_min = emg.min()
    emg_max = emg.max()
    emg = (emg - emg_min) / (emg_max - emg_min)
    plot_emg(emg[:, 0], plot_label + " after min-max")

    # low pass filter
    emg = lpf(emg)
    plot_emg(emg[:, 0], plot_label + " after lpf")

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
                # preprocess
                emg_left_preproc = preprocess_data(emg_left_data, "left")

                spectrogram.compute_spectrogram(emg_left_preproc[:, 0], "first miao")

                # # RIGHT
                for i in range(8):
                    # preprocess
                    emg_right_preproc = preprocess_data(emg_right_data[:, 1], "right")

                # add PREPROCESSED ACTIVATIONS to dictionary
                emg_dict["preprocessed_activation_left"] = emg_left_preproc
                emg_dict["preprocessed_activation_right"] = emg_right_preproc

                pkl_dict[emg_index] = emg_dict

            # # SAVE dictionary in pickle file
            # out_dir = "preprocessed_EMG"
            # output_path = os.path.join(out_dir, parsed_emg_filename.split('_')[0] + "_EMG")
            # pickle.dump(pkl_dict, open(str(output_path + ".pkl"), "wb"))


if __name__ == "__main__":
    main()
