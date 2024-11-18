from .DataPreparation import FrameSequenceDataset

main_folder_path = data_path

label_to_idx = find_classes(main_folder_path)[1]
dataset = FrameSequenceDataset(main_folder_path, label_to_idx, transform=data_transform)
frames, label = random.choice(dataset)
frames = frames.permute(1, 0, 2, 3)
frames