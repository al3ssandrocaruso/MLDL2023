"""
Script to parse HDF5 file into a pickle file:
the structure of the pickle file is a DICTIONARY with key=INDEX and value a DICTIONARY:
{ index:{
        "index": index of the action: given to match index in the pkl files in action-net folder
        "label": label of the action,
        "frames-indexes-array": array of integers
        "frames-timestamps-array": array of timestamps
        "start_time": start time in [s] of the activity
        "end_time": end time in [s] of the activity
        "duration": duration in seconds [s]
        }
    ...
}
"""
import os
import h5py
import numpy as np
import pickle

# Specify the directory to files to parse.
directory = "C:/Users/bracc/Desktop/hfd5_S_04"

####################################################
# Example of reading sensor data: read video data.
####################################################
def extract_frame_time_s(h5_file):
    device_name = 'eye-tracking-video-world'
    stream_name = 'frame_timestamp'
    # Get the timestamps for each row as seconds since epoch.
    frame_time_s = h5_file[device_name][stream_name]['time_s']
    frame_time_s = np.squeeze(np.array(frame_time_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list

    return frame_time_s

def extract_labels(hdf5_file_path):
    ####################################################
    # Reading label data
    ####################################################
    with h5py.File(hdf5_file_path, 'r') as h5_file:
        device_name = 'experiment-activities'
        stream_name = 'activities'

        # Get the timestamped label data.
        # As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
        activity_datas = h5_file[device_name][stream_name]['data']
        activity_times_s = h5_file[device_name][stream_name]['time_s']
        activity_times_s = np.squeeze(
            np.array(activity_times_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list
        # Convert to strings for convenience.
        activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]

        # Combine start/stop rows to single activity entries with start/stop times.
        #   Each row is either the start or stop of the label.
        #   The notes and ratings fields are the same for the start/stop rows of the label, so only need to check one.
        exclude_bad_labels = True  # some activities may have been marked as 'Bad' or 'Maybe' by the experimenter; submitted notes with the activity typically give more information
        activities_labels = []
        activities_start_times_s = []
        activities_end_times_s = []
        activities_ratings = []
        activities_notes = []
        for (row_index, time_s) in enumerate(activity_times_s):
            label = activity_datas[row_index][0]
            is_start = activity_datas[row_index][1] == 'Start'
            is_stop = activity_datas[row_index][1] == 'Stop'
            rating = activity_datas[row_index][2]
            notes = activity_datas[row_index][3]
            if exclude_bad_labels and rating in ['Bad', 'Maybe']:
                continue
            # Record the start of a new activity.
            if is_start:
                activities_labels.append(label)
                activities_start_times_s.append(time_s)
                activities_ratings.append(rating)
                activities_notes.append(notes)
            # Record the end of the previous activity.
            if is_stop:
                activities_end_times_s.append(time_s)
        return activities_labels, activities_start_times_s, activities_end_times_s

def main():
    ####################################################
    # Parsing every file.
    ####################################################
    for h5_filename in [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]:
        h5_filepath = os.path.join(directory, h5_filename)
        # Extract labels, timestamps for this file
        activities_labels, activities_start_times_s, activities_end_times_s = extract_labels(h5_filepath)
        # Open the file
        with h5py.File(h5_filepath, 'r') as h5_file:
            # dictionary to store the file
            pkl_dict = dict()
            ####################################################
            # Getting sensor data for each label.
            ####################################################
            for i in range(len(activities_start_times_s)):
                label_start_time_s = activities_start_times_s[i]
                label_end_time_s = activities_end_times_s[i]
                duration = label_end_time_s - label_start_time_s
                index = i + 1
                label = activities_labels[i]

                # LEFT myo
                frame_time_s = extract_frame_time_s(h5_file)

                frame_indexes_forLabel = np.where((frame_time_s >= label_start_time_s) & (frame_time_s <= label_end_time_s))[0]
                frame_timestamps_forLabel = frame_time_s[frame_indexes_forLabel]

                pkl_dict[index] = {
                    "index": index,
                    "label": label,
                    "frames-indexes-array": frame_indexes_forLabel,
                    "frames-timestamps-array": frame_timestamps_forLabel,
                    "start_time": label_start_time_s,
                    "end_time": label_end_time_s,
                    "duration": duration
                }

            ####################################################
            # Save file
            ####################################################
            out_dir = "parsed_FRAMES"
            base_name = os.path.basename(h5_filepath)  # get the filename component of the path
            output_path = os.path.join(out_dir, os.path.splitext(base_name)[0].split('_')[-1] + "_parsed_FRAMES.pkl")

            pickle.dump(pkl_dict, open(output_path, "wb"))


main()