import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from Data_Preparation.Data_class import FrameSequenceDataset, find_classes, data_transform
from Model_Class.Model_Class import ResNet101
from Training_Steps import train

device = "cuda" if torch.cuda.is_available() else "cpu"

main_folder_path = 'Frame Extraction/Argen_frames'
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', main_folder_path))

class_names, label_to_idx = find_classes(main_folder_path)
dataset = FrameSequenceDataset(main_folder_path, label_to_idx, transform=data_transform)

BATCH_SIZE = 2
NUM_WORKERS = 0

# Define the train-test split ratio (e.g., 80% train, 20% test)
train_size = int(0.80 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # Remaining 20% for testing

# Split the dataset into train and test sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for both sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

output_shape = len(class_names)

model_0 = ResNet101(img_channel=1, num_classes = output_shape).to(device)

output_shape = len(class_names)  # Number of classes

NUM_EPOCHS = 51

loss_fn = nn.CrossEntropyLoss()

learning_rate = 0.001
weight_decay = 1e-4

# Initialize Adam optimizer with weight decay
optimizer = torch.optim.Adam(model_0.parameters(),
                            lr=learning_rate,
                           weight_decay=weight_decay)

# Learning rate scheduler - Cosine Annealing for smoother decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

model_0_results = train(
    model = model_0,
    train_dataloader = train_loader,
    test_dataloader = test_loader,
    optimizer = optimizer,
    scheduler = scheduler,
    loss_fn = loss_fn,
    epochs = NUM_EPOCHS,
    model_save_folder="/models/",
    device = device
)