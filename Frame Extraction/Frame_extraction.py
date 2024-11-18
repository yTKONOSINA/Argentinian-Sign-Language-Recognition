import os
from pathlib import Path
from Frame_extraction_function import extract_frames

num_frames_to_extract = 20
video_paths = "../all_cut"

video_files_list = list(Path(video_paths).glob('*.mp4'))
video_files_list.sort()

for i, file_path in enumerate(video_files_list):

   # Get the filename without the directory
    filename = os.path.basename(file_path)  # This will give "001_001_001.mp4"

    #Split the filename at underscores and take the first part
    first_part = filename.split('_')[0]  # This will give "001"
    extract_frames(file_path, num_frames_to_extract, f"Argen_frames/{str(first_part)}/{i%50}")