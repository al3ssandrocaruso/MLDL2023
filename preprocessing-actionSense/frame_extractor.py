import os
import re

import cv2

dir_im = "C:/Users/bracc/Desktop/avi_files/images"
video_file = 'C:/Users/bracc/Desktop/avi_files/2022-06-14_16-38-43_S04_eye-tracking-video-worldGaze_frame.avi'

import cv2
import os

# specify the paths
input_video_path = "/path/to/input/video.avi"
output_folder_path = "/path/to/output/folder"

# specify the clips to extract
clip_frames = [[10, 20, 30], [50, 60, 70, 80], [100, 110]]

# create output folders for each clip
for i in range(len(clip_frames)):
    clip_folder_path = os.path.join(output_folder_path, f"clip_{i+1}")
    os.makedirs(clip_folder_path, exist_ok=True)

# extract the frames for each clip and save them in the corresponding folder
cap = cv2.VideoCapture(input_video_path)
frame_index = 0
clip_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_index in clip_frames[clip_index]:
        clip_folder_path = os.path.join(output_folder_path, f"clip_{clip_index+1}")
        output_path = os.path.join(clip_folder_path, f"frame_{frame_index}.png")
        cv2.imwrite(output_path, frame)
    frame_index += 1
    if frame_index == clip_frames[clip_index][-1] + 1:
        clip_index += 1
cap.release()
