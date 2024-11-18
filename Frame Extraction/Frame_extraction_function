import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def extract_frames(video_path, num_frames, output_folder, edge=0, start_frame=None, end_frame=None):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # If start_frame and end_frame are not defined, calculate them based on edge
    if start_frame is None:
        start_frame = edge
    if end_frame is None:
        end_frame = total_frames - edge

    # Validate the start_frame and end_frame
    if start_frame < 0 or end_frame > total_frames or start_frame >= end_frame:
        print("Invalid start or end frame. Adjusting to fit within video length.")
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

    # Calculate the available frames range
    frame_range = end_frame - start_frame

    # Extract frames from the video
    frames = []
    count = 0

    while video.isOpened() and start_frame + count < end_frame:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame + count)
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
        count += frame_range // num_frames if frame_range >= num_frames else 1

    video.release()

    # Interpolate if not enough frames were collected
    interpolated_frames = []

    if len(frames) < num_frames:
        for i in range(len(frames) - 1):
            interpolated_frames.append(frames[i])  # Add the current frame
            # Generate interpolated frames between frames[i] and frames[i+1]
            for j in range(1, num_frames // (len(frames) - 1) + 1):
                alpha = j / (num_frames // (len(frames) - 1) + 1)
                interpolated_frame = cv2.addWeighted(frames[i], 1 - alpha, frames[i + 1], alpha, 0)
                interpolated_frames.append(interpolated_frame)
        interpolated_frames.append(frames[-1])  # Add the last frame
    else:
        interpolated_frames = frames

    # If we still don't have enough frames after interpolation, add more
    while len(interpolated_frames) < num_frames:
        last_frame = interpolated_frames[-1]
        interpolated_frames.append(last_frame)  # Repeat last frame to fill the gap

    # Save the frames (original and interpolated) with transformations
    for i, frame in enumerate(interpolated_frames[:num_frames]):
        # Convert frame (which is a NumPy array) to a PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Apply transformations to the frame
        transformed_frame = data_transform(pil_image)
        
        # Convert the tensor back to a NumPy array for saving
        transformed_frame = transformed_frame.permute(1, 2, 0).numpy()  # Change the order for saving

        # Convert the tensor to uint8 for saving with OpenCV
        transformed_frame = (transformed_frame * 255).astype('uint8')

        # Save the transformed frame
        frame_filename = os.path.join(output_folder, f"frame_{i}.jpg")
        cv2.imwrite(frame_filename, cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR))

    print(f"Extracted and transformed {len(interpolated_frames[:num_frames])} frames to {output_folder}.")
