import numpy as np
import os
import shutil
import pickle
from PIL import Image

arrays_of_indices_in = []
with open(file="./SUB_spectrogram/S04_spectrogram.pkl", mode='rb') as preprocessed_emg:
    for pre_index, pre_dict in pickle.load(preprocessed_emg).items():
        frames_sub_indexes = pre_dict["frames-sub_indexes-array"]
        arrays_of_indices_in.append(frames_sub_indexes)

#print(arrays_of_indices_in)


# Specify the path to the directory where the frames are extracted
frames_directory_in = "/Users/alessandrocaruso/Downloads"
frames_directory_out = "/Users/alessandrocaruso/Downloads/frames"


def stretch_image(image_path, new_width, new_height):
    # Open the image file
    image = Image.open(image_path)

    # Resize the image using the scaling factors and NEAREST resampling
    stretched_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Return the stretched image
    return stretched_image
def resize_images_in_folder(folder_path, new_width, new_height):
    count=0
    # Iterate through all files and subdirectories in the folder
    for root, _, files in os.walk(folder_path):
        if(root=="/Users/alessandrocaruso/Downloads/frames"): continue
        count+=1
        for file in files:
            # Check if the file is an image
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Get the file path
                file_path = os.path.join(root, file)

                # Resize the frame while maintaining the center
                resized_frame = stretch_image(file_path, new_width, new_height)

                # Delete the original file
                os.remove(file_path)

                # Save the resized frame in the same folder
                resized_frame.save(file_path)
        print("iteration: ", count, "\nsubdir: ", root)

resize_images_in_folder(frames_directory_out, 456, 256)

def move_frames(out_dir=frames_directory_out, frames_directory=frames_directory_in, arrays_of_indices=arrays_of_indices_in):
    # create list for all filenames
    all_filenames = []
    for array_of_index in arrays_of_indices:
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
            shutil.move(source_path, out_destination_path)
            #shutil.copy(source_path, out_destination_path)

            # print(f"Moved {filename} to {out_folder_name}")

