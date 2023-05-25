import numpy as np
import os
import shutil
import pickle

########################################################################################################################################
arrays_of_indices_in = []
# with open(file="./SUB_spectrogram/S04_spectrogram.pkl", mode='rb') as preprocessed_emg:
#     for pre_index, pre_dict in pickle.load(preprocessed_emg).items():
#         frames_sub_indexes = pre_dict["frames-sub_indexes-array"]
#         label = pre_dict["label"]
#         arrays_of_indices_in.append(frames_sub_indexes)
#         print(pre_index)
#         print()
# print(arrays_of_indices_in)

# Specify the path to the directory where the frames are extracted
frames_directory_in = "C:\\Users\\bracc\\Desktop\\2008_graz"
frames_directory_out = "MIAO"
########################################################################################################################################
def move_frames(out_dir=frames_directory_out, frames_directory=frames_directory_in, arrays_of_indices=arrays_of_indices_in):
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

# move_frames()
########################################################################################################################################
import os
import cv2

def resize_with_center(frame, new_width, new_height):
    # Get the original frame dimensions
    height, width = frame.shape[:2]

    # Calculate the center coordinates of the original frame
    center_x = width // 2
    center_y = height // 2

    # Calculate the starting and ending coordinates for cropping
    start_x = center_x - (new_width // 2)
    end_x = start_x + new_width
    start_y = center_y - (new_height // 2)
    end_y = start_y + new_height

    # Crop the frame using the calculated coordinates
    cropped_frame = frame[start_y:end_y, start_x:end_x]

    # Resize the cropped frame to the desired dimensions
    resized_frame = cv2.resize(cropped_frame, (new_width, new_height))

    return resized_frame

def resize_images_in_folder(folder_path, new_width, new_height):
    # Iterate through all files and subdirectories in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an image
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Get the file path
                file_path = os.path.join(root, file)

                # Load the input frame
                input_frame = cv2.imread(file_path)

                # Resize the frame while maintaining the center
                resized_frame = resize_with_center(input_frame, new_width, new_height)

                # # Delete the original file
                # os.remove(file_path)

                # Save the resized frame in the same folder
                # cv2.imwrite(file_path, resized_frame)
                # TEST
                basename = file_path.split(".")[0]
                basename = str(str(basename) + str("_resized"))
                file_path = str(basename + ".png")
                cv2.imwrite(file_path, resized_frame)
        print("Fatto una volta")

# Specify the path to the root folder containing subfolders with images
root_folder_path = frames_directory_in

# Resize images in all subfolders under the root folder
resize_images_in_folder(root_folder_path, 456, 256)
