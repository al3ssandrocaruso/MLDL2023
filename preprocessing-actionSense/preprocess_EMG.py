"""
Structure of HDF5 file: similar to a dictionary of dictionaries of dictionaries... (key: value), same way to access
first keys: ['experiment-activities', 'experiment-calibration', 'experiment-notes', 'eye-tracking-gaze', 'eye-tracking-pupil', 'eye-tracking-time', 'eye-tracking-video-eye', 'eye-tracking-video-world', 'eye-tracking-video-worldGaze', 'myo-left', 'myo-right', 'tactile-calibration-scale', 'tactile-glove-left', 'tactile-glove-right', 'xsens-CoM', 'xsens-ergonomic-joints', 'xsens-foot-contacts', 'xsens-joints', 'xsens-segments', 'xsens-sensors', 'xsens-time']>
'myo-left', 'myo-right' keys: ['acceleration_g', 'angular_velocity_deg_s', 'battery', 'emg', 'gesture', 'orientation_quaternion', 'rssi', 'synced']
'emg' keys: ['data', 'time_s', 'time_s_original', 'time_str']
"""
import pickle
import numpy as np
import os

# specify directory of the file to preprocess
directory = "./parsed_EMG"


def plot_emg(emg, channel, fs=160):
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
    ax.set_title('EMG Signal: {}'.format(channel))

    plt.show()

def low_pass(emg, order=5, cutoff_freq=0.5, fs=160):
    from scipy.signal import butter, filtfilt

    # Define the filter coefficients
    b, a = butter(N=order, Wn=cutoff_freq, fs=fs, btype='lowpass', analog=False, output='ba')
    # Reshape
    emg = emg.reshape(-1)
    # Apply the filter
    emg = filtfilt(b, a, emg)

def preprocess_data(emg, plot_label):
    # abs value ALREADY DONE IN PARSE
    # df = df.abs()

    # low pass filter
    low_pass(emg)

    # min-max [0, 1] normalization
    emg_min = emg.min()
    emg_max = emg.max()
    emg = (emg - emg_min) / (emg_max - emg_min)

    # Plot of emg signal after preprocessing
    plot_emg(emg, plot_label)


def main():
    for parsed_emg_filename in [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]:
        parsed_emg_filepath = os.path.join(directory, parsed_emg_filename)
        with open(parsed_emg_filepath, 'rb') as parsed_emg:
            for emg_dict in pickle.load(parsed_emg).values():
                emg_left_act = emg_dict['myo-left-activation']
                emg_right_act = emg_dict['myo-right-activation']

                left_act_df = np.array(emg_left_act)
                preprocess_data(left_act_df, "left")

                right_act_df = np.array(emg_right_act)
                preprocess_data(right_act_df, "right")

                # SAVE FILE
                out_dir = "preprocessed"
                output_path = os.path.join(out_dir, parsed_emg_filename.split('_')[0])
                pickle.dump(pkl_dict, open(str(output_path + ".pkl"), "wb"))

if __name__ == "__main__":
    main()
