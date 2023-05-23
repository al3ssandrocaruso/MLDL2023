import os
import pickle
import subprocess
import numpy as np

# Specify the paths
input_pickle_dir= "./parsed_FRAMES"
output_folder_dir = "C:/Users/bracc/Desktop/avi_files/images"
input_video_path = 'C:/Users/bracc/Desktop/avi_files/2022-06-14_16-38-43_S04_eye-tracking-video-world_frame.mp4'

"""
Extract frame indexes for every action, append to a list
"""
clip_frames = []
# Process each file in the input_pickle_path directory
for parsed_emg_filename in os.listdir(input_pickle_dir):
    parsed_emg_filepath = os.path.join(input_pickle_dir, parsed_emg_filename)
    if os.path.isfile(parsed_emg_filepath):
        with open(parsed_emg_filepath, 'rb') as parsed_emg:
            for emg_index, emg_dict in pickle.load(parsed_emg).items():
                clip_frames.append(list(emg_dict['frames-indexes-array']))
print(clip_frames)

def count_elements(lst):
    count = 0
    for sub_list in lst:
        count += len(sub_list)
    return count

total_elements = count_elements(clip_frames)
print(total_elements)

exit(0)
"""
Create output folders for each clip
"""
for i in range(len(clip_frames)):
    clip_folder_path = os.path.join(output_folder_dir, f"S04_{i + 1}")
    os.makedirs(clip_folder_path, exist_ok=True)

def extract_frames(output_folder, frame_indices, input_video):
    """
    EXTRACT and SAVE frames
    Args:
        output_folder: folder to save every frame
        frame_indices: (list of int), frames we want to extract
        input_video: path to MP4 video, constant
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate the select filter expression
    select_expression = '+'.join([f'eq(n,{index})' for index in frame_indices])

    # Construct the FFmpeg command
    command = [
        'ffmpeg',
        '-i', input_video,
        '-vf', f'select=\'{select_expression}\',setpts=N/FRAME_RATE/TB',
        os.path.join(output_folder, f'frame_%010d.png')
    ]

    print(command)

    # Execute the command
    # subprocess.run(["cmd.exe", "echo", "ciao"])
    subprocess.run(command)


# Example usage
output_folder_path = os.path.join(output_folder_dir, "first_try")
print(output_folder_path)

partitions = np.array_split(clip_frames[0], 5)
for partition in partitions[:1]:
    partition = list(partition)
    print(partition)
    extract_frames(output_folder_path, frame_indices=partition, input_video=input_video_path)
