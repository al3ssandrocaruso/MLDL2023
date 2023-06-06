"""
script to divide longer action in smaller sub-action with the SAME label
"""
import os
import pickle
import math
import numpy as np
import spectrogram
import spectrogram_clip

FRAME_RATE = 30
SECONDS = 5
LENGTH = math.floor(FRAME_RATE * SECONDS)

INTERP_LENGTH = 800  # points
N_CLIPS = 5
# not used
CUT_PAD_LENGTH = 800


# frame_pickle_path = "./parsed_FRAMES/S04_parsed_FRAMES.pkl"
# emg_pickle_path = "./preprocessed_EMG/S04_EMG.pkl"


# minimum number of frames: 81 (around 3 seconds)

def split_arrays(indexes, timestamps):
    """
    divide indexes and timestamps arrays into sub-arrays
    returns:
        - sub_indexes: list of sub-action array
        - ... same ...
    """
    sub_indexes = []
    sub_timestamps = []
    i = 0
    while i + LENGTH < indexes.shape[0]:
        subarray1 = indexes[i: i + LENGTH]
        subarray2 = timestamps[i: i + LENGTH]
        sub_indexes.append(np.array(subarray1))
        sub_timestamps.append(np.array(subarray2))
        i += LENGTH

    return sub_indexes, sub_timestamps


def cut_and_pad(emg_left_sub, emg_right_sub):
    """
    cut or pad with 0 to reach CUT_PAD_LENGTH
    why 800?
    left avg 799.7619047619048
    right avg 795.7394957983194
    (avg only for action longer than 5 seconds)
    """
    # Cut or pad emg_left_sub
    if emg_left_sub.shape[0] > CUT_PAD_LENGTH:
        emg_left_sub = emg_left_sub[:CUT_PAD_LENGTH, :]
    elif emg_left_sub.shape[0] < CUT_PAD_LENGTH:
        padding = ((0, CUT_PAD_LENGTH - emg_left_sub.shape[0]), (0, 0))
        emg_left_sub = np.pad(emg_left_sub, padding, mode='constant')

    # Cut or pad emg_right_sub
    if emg_right_sub.shape[0] > CUT_PAD_LENGTH:
        emg_right_sub = emg_right_sub[:CUT_PAD_LENGTH, :]
    elif emg_right_sub.shape[0] < CUT_PAD_LENGTH:
        padding = ((0, CUT_PAD_LENGTH - emg_right_sub.shape[0]), (0, 0))
        emg_right_sub = np.pad(emg_right_sub, padding, mode='constant')

    return emg_left_sub, emg_right_sub


def interpolate(emg_left_sub, emg_right_sub):
    """
    cut or pad with 0 to reach CUT_PAD_LENGTH
    why 800?
    left avg 799.7619047619048
    right avg 795.7394957983194
    (avg only for action longer than 5 seconds)
    """
    # # LEFT
    # Create an empty array to store the interpolated channels
    interp_emg_left_sub = np.empty((INTERP_LENGTH, 8))
    for i in range(8):
        channel = emg_left_sub[:, i]  # Extract the ith channel
        x = np.arange(len(channel))  # x-coordinates for interpolation
        new_x = np.linspace(0, len(channel) - 1, INTERP_LENGTH)  # New x-coordinates for interpolation
        interpolated_channel = np.interp(new_x, x, channel)  # Interpolate the channel
        interp_emg_left_sub[:, i] = interpolated_channel  # Store the interpolated channel

    # # RIGHT
    # Create an empty array to store the interpolated channels
    interp_emg_right_sub = np.empty((INTERP_LENGTH, 8))
    for i in range(8):
        channel = emg_right_sub[:, i]  # Extract the ith channel
        x = np.arange(len(channel))  # x-coordinates for interpolation
        new_x = np.linspace(0, len(channel) - 1, INTERP_LENGTH)  # New x-coordinates for interpolation
        interpolated_channel = np.interp(new_x, x, channel)  # Interpolate the channel
        interp_emg_right_sub[:, i] = interpolated_channel  # Store the interpolated channel

    return interp_emg_left_sub, interp_emg_right_sub


def divide_array_into_clips(arr, n_clips):
    clip_size = len(arr) // n_clips
    clips = np.split(arr[:clip_size * n_clips], n_clips, axis=0)
    return clips


def frames_timestamps(frame_pickle_path):
    """
    divide frame array of an action into subarrays
    returns
        - sub_action_dict: dictionary of dictionaries, every dictionary is a SUB-ACTION:
            {frames-sub_indexes-array: np.array,
            frames-sub_timestamps_array: np.array,
            label: str}
        - sub_timestamps_list: list of sub-timestamps arrays
    index is preserved
    """
    with open(frame_pickle_path, 'rb') as parsed_frames:
        sub_timestamps_list = list()
        sub_action_dict = dict()
        index = 0
        for _, emg_dict in pickle.load(parsed_frames).items():
            indexes = emg_dict['frames-indexes-array']
            timestamps = emg_dict['frames-timestamps-array']
            label = emg_dict["label"]

            # Split action in sub-actions
            if emg_dict["duration"] >= SECONDS:  # if duration>=5s -> split
                sub_indexes, sub_timestamps = split_arrays(indexes, timestamps)
            else:  # else no split
                sub_indexes = [indexes]
                sub_timestamps = [timestamps]

            # general list to keep sub_timestamps
            sub_timestamps_list.append(sub_timestamps)

            # add each sub-action to a global_list
            for i in range(len(sub_indexes)):
                sub_action_dict[index] = {
                    "index": index,
                    "label": label,
                    "frames-sub_indexes-array": sub_indexes[i],
                    "frames-sub_timestamps_array": sub_timestamps[i]
                }
                index += 1
    return sub_action_dict, sub_timestamps_list


def plot_emg(signal, title_plot, sr=160):
    """
    Plots an EMG signal.

    Args:
        signal (list or numpy array): EMG signal values.
        sr (float): Sampling rate of the signal (samples per second).
    """
    import matplotlib.pyplot as plt
    # Calculate the time axis based on the signal length and sampling rate
    duration = len(signal) / sr
    time = [t / sr for t in range(len(signal))]
    # Create the plot
    plt.figure()
    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title_plot)
    plt.grid(True)
    plt.show()


def emg_timestamps(sub_action_dict, sub_timestamps_list, emg_pickle_path, emg_pickle_filename):
    with open(emg_pickle_path, 'rb') as preprocessed_emg:
        pkl_dict_CLIP = dict()
        pkl_dict = dict()
        index = 0  # [0-365]
        for pre_index, pre_dict in pickle.load(preprocessed_emg).items():
            left_pre_processed_data = pre_dict["preprocessed-left-data"]
            right_pre_processed_data = pre_dict["preprocessed-right-data"]
            left_timestamps = pre_dict["myo-left-timestamps"]
            right_timestamps = pre_dict["myo-right-timestamps"]
            label = pre_dict["label"]

            # print(f"index[{pre_index}]")
            # print(f"#sub_timestramps:{len(sub_timestamps_list[pre_index])}")

            # for every sub_timestamp -> split emg data
            for sub_timestamp in sub_timestamps_list[pre_index]:
                # get start and end time
                sub_label_start_time = sub_timestamp[0]
                sub_label_end_time = sub_timestamp[-1]

                # # LEFT
                emg_indexes_left = \
                    np.where((left_timestamps >= sub_label_start_time) & (left_timestamps <= sub_label_end_time))[0]
                emg_left_sub_data = left_pre_processed_data[emg_indexes_left, :]

                # # RIGHT
                emg_indexes_right = \
                    np.where((right_timestamps >= sub_label_start_time) & (right_timestamps <= sub_label_end_time))[0]
                emg_right_sub_data = right_pre_processed_data[emg_indexes_right, :]

                # print("BEFORE cut_pad/interp L", emg_left_sub_data.shape)
                # print("BEFORE cut_pad/interp R", emg_right_sub_data.shape)
                # plot_emg(emg_left_sub_data[:, 0], "BEFORE Left")

                # # CUT or PAD left right to have same length
                # emg_left_sub_padded, emg_right_sub_padded = cut_and_pad(emg_left_sub=emg_left_sub_data,
                #                                                         emg_right_sub=emg_right_sub_data)

                # INTERPOLATE
                emg_left_sub_pad_interp, emg_right_sub_pad_interp = interpolate(emg_left_sub=emg_left_sub_data,
                                                                                emg_right_sub=emg_right_sub_data)

                # print("AFTER cut_pad/interp L", emg_left_sub_pad_interp.shape)
                # print("AFTER cut_pad/interp R", emg_right_sub_pad_interp.shape)
                # plot_emg(emg_left_sub_pad_interp[:, 0], "AFTER Left")
                # print()
                pkl_dict[index] = {
                    "index": index,
                    "label": label,
                    "frames-sub_indexes-array": sub_action_dict[index]["frames-sub_indexes-array"],
                    "preprocessed-left-data": emg_left_sub_pad_interp,
                    "preprocessed-right-data": emg_right_sub_pad_interp,
                    "start_time": sub_label_start_time,
                    "end_time": sub_label_end_time,
                    "duration": sub_label_end_time - sub_label_start_time
                }

                ################################### CLIP ###########################################

                left_clips = divide_array_into_clips(emg_left_sub_pad_interp, n_clips=N_CLIPS)
                rigth_clips = divide_array_into_clips(emg_right_sub_pad_interp, n_clips=N_CLIPS)

                pkl_dict_CLIP[index] = {
                    "index": index,
                    "label": label,
                    "frames-sub_indexes-array": sub_action_dict[index]["frames-sub_indexes-array"],
                    "preprocessed-left-clip-list": left_clips,
                    "preprocessed-right-clip-list": rigth_clips,
                    "start_time": sub_label_start_time,
                    "end_time": sub_label_end_time,
                    "duration": sub_label_end_time - sub_label_start_time
                }
                # print("L", len(left_clips))
                # print("L[0]:", left_clips[0].shape)
                #
                # print("R", len(rigth_clips))
                # print("R[0]:", rigth_clips[0].shape)
                ######################################################################################
                # increment index for every sub action
                index += 1

        # SAVE dictionary in pickle file
        out_dir = "SUB_preprocessed"
        output_path = os.path.join(out_dir, emg_pickle_filename.split('_')[0] + "_SUB")
        pickle.dump(pkl_dict, open(str(output_path + ".pkl"), "wb"))
        print(f"DONE dumping SUB_preprocessed in {output_path}.pkl")

        # SAVE dictionary in pickle file
        out_dir = "SUB_CLIP_preprocessed"
        output_path = os.path.join(out_dir, emg_pickle_filename.split('_')[0] + "_SUB_CLIP")
        pickle.dump(pkl_dict, open(str(output_path + ".pkl"), "wb"))
        print(f"DONE dumping SUB_CLIP_preprocessed in {output_path}.pkl")

        # # SAVE dictionary in pickle file
        # out_dir = "SUB_preprocessed"
        # output_path = os.path.join(out_dir, "S04_SUB.pkl")
        # pickle.dump(pkl_dict, open(output_path, "wb"))

        # # SAVE CLIP
        # out_dir = "SUB_CLIP_preprocessed"
        # output_path = os.path.join(out_dir, "S04_SUB_CLIP.pkl")
        # pickle.dump(pkl_dict_CLIP, open(output_path, "wb"))

        print(f"DONE file: {emg_pickle_filename}")
        print()


def main():
    directory = "./parsed_FRAMES"
    for frame_pickle_filename in [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]:
        split = frame_pickle_filename.split('_')[0]
        frame_pickle_path = os.path.join(directory, frame_pickle_filename)

        # compute SUB-ACTIONS
        sub_action_dict, sub_timestamps_list = frames_timestamps(frame_pickle_path)

        pre_dir = "./preprocessed_EMG"
        emg_pickle_filename = str(split + "_EMG.pkl")
        emg_pickle_path = os.path.join(pre_dir, emg_pickle_filename)

        # compute and save EMG subactions and clips
        emg_timestamps(sub_action_dict, sub_timestamps_list, emg_pickle_path, emg_pickle_filename)
    print()
    print("DONE Preprocessing SUB-CLIP")
    print()
    print()
    #######################################################################################
    # # NORMAL spectrograms
    # input_file = "./SUB_preprocessed/S04_SUB.pkl"
    # out_dir = "SUB_spectrogram"
    # output_path = os.path.join(out_dir, "S04_spectrogram")
    # spectrogram.spectrogram_main(input_file, output_path, isSubaction=True)

    in_dir = "./SUB_preprocessed"
    out_dir = "./SUB_spectrogram"
    for sub_pre_filename in [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]:
        input_file = os.path.join(in_dir, sub_pre_filename)
        output_path = os.path.join(out_dir, sub_pre_filename.split('_')[0] + "_spectrogram")

        spectrogram.spectrogram_main(input_file, output_path, isSubaction=True)
        print(f"Done SPECTROGRAMS of {sub_pre_filename}")

    print("Done NORMAL")
    #######################################################################################
    # # CLIP spectrograms
    # input_file_CLIP = "./SUB_CLIP_preprocessed/S04_SUB_CLIP.pkl"
    # out_dir_CLIP = "SUB_CLIP_spectrograms"
    # output_path_CLIP = os.path.join(out_dir_CLIP, "S04_spectrogram_CLIP")
    # spectrogram_clip.spectrogram_main_CLIP(input_file_CLIP, output_path_CLIP, n_clips=5, isSubaction=True)

    in_dir = "./SUB_CLIP_preprocessed"
    out_dir = "./SUB_CLIP_spectrograms"
    for sub_pre_filename in [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]:
        input_file_CLIP = os.path.join(in_dir, sub_pre_filename)
        output_path_CLIP = os.path.join(out_dir, sub_pre_filename.split('_')[0] + "_spectrogram_CLIP")

        print("input_file_CLIP", input_file_CLIP)
        print("output_path_CLIP", output_path_CLIP)
        print()

        spectrogram_clip.spectrogram_main_CLIP(input_file_CLIP, output_path_CLIP, n_clips=5, isSubaction=True)

    print("Done CLIP")


if __name__ == "__main__":
    main()
