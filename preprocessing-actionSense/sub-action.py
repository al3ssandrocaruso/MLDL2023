"""
script to divide longer action in smaller sub-action with the SAME label
"""
import os
import pickle
import math
import numpy as np
import spectrogram

FRAME_RATE = 30
SECONDS = 5
LENGTH = math.floor(FRAME_RATE * SECONDS)
frame_pickle_path = "./parsed_FRAMES/S04_parsed_FRAMES.pkl"
emg_pickle_path = "./preprocessed_EMG/S04_EMG.pkl"


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
    # Calculate the average length
    CUT_PAD_LENGTH = 800

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


def frames_timestamps():
    """
    divide frame array of an action into subarrays
    returns
        - magic_list: list of dictionaries, every dictionary is a SUB-ACTION:
            {frames-sub_indexes-array: np.array,
            frames-sub_timestamps_array: np.array,
            label}
        - sub_timestamps_list: list of sub-timestamps arrays
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
            if emg_dict["duration"] >= SECONDS:  # if duration >= 5s -> split
                sub_indexes, sub_timestamps = split_arrays(indexes, timestamps)
            else:  # else pad original array to LENGTH
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

def emg_timestamps(sub_action_dict, sub_timestamps_list):
    with open(emg_pickle_path, 'rb') as preprocessed_emg:
        pkl_dict = dict()
        index = 0  # [0-365]
        for pre_index, pre_dict in pickle.load(preprocessed_emg).items():
            left_pre_processed_data = pre_dict["preprocessed-left-data"]
            right_pre_processed_data = pre_dict["preprocessed-right-data"]
            left_timestamps = pre_dict["myo-left-timestamps"]
            right_timestamps = pre_dict["myo-right-timestamps"]
            label = pre_dict["label"]

            # for every sub_timestamp -> split emg data
            for sub_timestamp in sub_timestamps_list[pre_index]:
                # get start and end time
                sub_label_start_time = sub_timestamp[0]
                sub_label_end_time = sub_timestamp[-1]

                # # LEFT
                emg_indexes_left = np.where((left_timestamps >= sub_label_start_time) & (left_timestamps <= sub_label_end_time))[0]
                emg_left_sub_data = left_pre_processed_data[emg_indexes_left, :]

                # # RIGHT
                emg_indexes_right = np.where((right_timestamps >= sub_label_start_time) & (right_timestamps <= sub_label_end_time))[0]
                emg_right_sub_data = right_pre_processed_data[emg_indexes_right, :]

                # CUT or PAD left right to have same length
                emg_left_sub_padded, emg_right_sub_padded = cut_and_pad(emg_left_sub=emg_left_sub_data,
                                                                        emg_right_sub=emg_right_sub_data)
                print("emg_left_sub_padded", emg_left_sub_padded.shape)
                print("emg_right_sub_padded", emg_right_sub_padded.shape)
                pkl_dict[index] = {
                    "index": index,
                    "label": label,
                    "frames-sub_indexes-array": sub_action_dict[index]["frames-sub_indexes-array"],
                    "preprocessed-left-data": emg_left_sub_padded,
                    "preprocessed-right-data": emg_right_sub_padded,
                    "start_time": sub_label_start_time,
                    "end_time": sub_label_end_time,
                    "duration": sub_label_end_time-sub_label_start_time
                }

                # increment index for every sub action
                index += 1

        # SAVE dictionary in pickle file
        out_dir = "SUB_preprocessed"
        output_path = os.path.join(out_dir, "S04_SUB.pkl")
        pickle.dump(pkl_dict, open(output_path, "wb"))


def main():
    sub_action_dict, sub_timestamps_list = frames_timestamps()

    emg_timestamps(sub_action_dict, sub_timestamps_list)

    input_file = "./SUB_preprocessed/S04_SUB.pkl"
    out_dir = "SUB_spectrogram"
    output_path = os.path.join(out_dir, "S04_spectrogram")
    spectrogram.spectrogram_main(input_file, output_path, True)


if __name__ == "__main__":
    main()