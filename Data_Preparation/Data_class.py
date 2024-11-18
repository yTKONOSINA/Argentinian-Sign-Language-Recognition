import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from typing import List, Dict, Tuple

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

class FrameSequenceDataset(Dataset):
    def __init__(self, main_folder, label_to_idx, transform=None, augmentation = None):
        self.main_folder = Path(main_folder)
        self.transform = transform
        self.label_to_idx = label_to_idx
        self.augmentation = augmentation
        # Get a list of all (subclass, index) pairs
        self.sequence_paths = []
        self.labels = []

        for subclass in self.main_folder.iterdir():
            if subclass.is_dir():
                for index_folder in subclass.iterdir():
                    if index_folder.is_dir():
                        self.sequence_paths.append(index_folder)
                        self.labels.append(subclass.name)

    def video_transform(self, video: torch.Tensor, angle_range: tuple = (-10, 10), translate_range: tuple = (0, 15), shear_range: tuple = (-10, 10)):
        # Define a transformation using `transforms.functional.affine`
        angle = random.uniform(*angle_range)           # Random angle within angle_range
        translate = (random.uniform(-translate_range[0], translate_range[0]),  # Random x-translation
                    random.uniform(-translate_range[1], translate_range[1]))  # Random y-translation
        shear = random.uniform(*shear_range)  
        def apply_transform(image):
            return transforms.functional.affine(
                image,
                angle=angle,            # Rotate by the specified angle
                translate=translate,     # Translate by specified (x, y) offsets
                scale=1.0,               # No scaling applied
                shear=[shear, shear]     # Shear by specified angle in both x and y directions
            )
        
        # Apply the transformation to each image in the video tensor
        transformed_video = torch.stack([apply_transform(image) for image in video])
        
        return transformed_video

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        sequence_folder = self.sequence_paths[idx]
        label = self.labels[idx]

        # Convert label to index using the label_to_idx dictionary
        label_idx = self.label_to_idx.get(label, -1)  # Use -1 for unknown labels

        # Load all frame images in the sequence
        frame_files = sorted(sequence_folder.glob('*.jpg'))  # Assuming frames are saved as .jpg
        frames = [Image.open(frame_file) for frame_file in frame_files]

        # Apply transformations to each frame
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        
        frames = self.video_transform(frames)

        return frames.permute(1, 0, 2, 3), label_idx  # Return frames tensor and the label index
    
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folder names in a target directory.
    """

    print(directory)

    # Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # Raise an error if class names could not be found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}... please check file strucure")

    # Create a dictionary of index labels
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx