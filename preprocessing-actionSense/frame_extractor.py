import numpy as np
import os
import shutil
import pickle

arrays_of_indices_in = []
with open(file="./SUB_spectrogram/S04_spectrogram.pkl", mode='rb') as preprocessed_emg:
    for pre_index, pre_dict in pickle.load(preprocessed_emg).items():
        frames_sub_indexes = pre_dict["frames-sub_indexes-array"]
        arrays_of_indices_in.append(frames_sub_indexes)

print(arrays_of_indices_in)


# Specify the path to the directory where the frames are extracted
frames_directory_in = "C:\\Users\\bracc\\Desktop\\2008_graz"


def move_frames(out_dir=frames_directory_in, frames_directory=frames_directory_in, arrays_of_indices=arrays_of_indices_in):
    # create list for all filenames
    all_filenames = []
    for array_of_index in arrays_of_indices[:1]:
        filenames = ["frame_%010d.png" % index for index in array_of_index]

        all_filenames.append(filenames)

    # Create a folder for each array and move the corresponding images
    for action_index, file_array in enumerate(all_filenames):
        # Create a folder for the current array
        out_folder_name = f'S04_{action_index+1:03d}'
        out_folder_path = os.path.join(out_dir, out_folder_name)
        os.makedirs(out_folder_path, exist_ok=True)

        # Iterate over the indices in the current array
        for filename in file_array:
            # Move the image file to the corresponding folder
            source_path = os.path.join(frames_directory, filename)
            out_destination_path = os.path.join(out_folder_path, filename)
    #
    # COPY OR MOVE
    #
            # shutil.move(source_path, destination_path)
            shutil.copy(source_path, out_destination_path)

            # print(f"Moved {filename} to {out_folder_name}")

move_frames()