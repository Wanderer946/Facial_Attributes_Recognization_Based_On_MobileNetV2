import torch
from PIL import Image
import pandas as pd
import yaml
import os
from torch.utils.data import DataLoader
from torchvision import transforms

class AttributeDatasetLoader:
    def __init__(self):
        self.df = None
        self.invalid_lines = []

    def load(self, label_path, selected_attrs=None):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Get the columns
        attribute_names = lines[0].strip().split()
        expected_length = len(attribute_names)

        # Check the number of fields in each row
        data = []
        for i, line in enumerate(lines[1:], start=2):
            parts = line.strip().split()
            if len(parts) != expected_length:
                self.invalid_lines.append((i, len(parts), expected_length, line.strip()))
                continue
            filename = parts[0]
            labels = list(map(int, parts[1:]))
            data.append([filename] + labels)

        # Build the DataFrame（include filename）
        self.df = pd.DataFrame(data, columns=attribute_names)

        # Check mistake
        if self.invalid_lines:
            print("The number of fields in the following lines does not match the title line:")
            for line_num, actual, expected, content in self.invalid_lines:
                print(f"Row: {line_num} Col: {actual}, Expected: {expected}, Content: {content}")
        # else:
        #     print("All data normal")

        # Only retain the specified attribute column (if any)
        if selected_attrs:
            keep_columns = selected_attrs
            self.df = self.df[keep_columns]

        return self.df
    

class ImageDatasetLoader:
    def __init__(self, dataframe, IMG_DIR, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = IMG_DIR
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'Filename']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        label = self.df.loc[idx].drop('Filename').astype(int).tolist()  # drop the Filename column
        label = torch.tensor(label, dtype=torch.float32)  # Transform into tensor

        if self.transform:
            image = self.transform(image)

        return image, label

def Load_Dataset(LABEL_PATH, IMG_DIR, PARTITION_PATH):
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    BATCH_SIZE = config['BATCH_SIZE']
    IMAGE_SIZE =  config['IMAGE_SIZE']
    NUM_WORKERS = config['NUM_WORKERS']

    # 1. Load label
    loader = AttributeDatasetLoader()
    target_attrs = ['Filename', 'Black_Hair', 'Eyeglasses', 'Heavy_Makeup', 'Male', 'Mustache',
                    'Pale_Skin', 'Smiling', 'Wavy_Hair', 'Wearing_Hat', 'Young']
    df = loader.load(LABEL_PATH, target_attrs)
    print(df.head())

    # 2. Split partition
    partition_map = {}
    with open(PARTITION_PATH, 'r') as f:
        for line in f:
            fname, label = line.strip().split()
            partition_map[fname] = int(label)

    df['Partition'] = df['Filename'].map(partition_map)

    # 3. Define transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # 4. Build datasets
    train_df = df[df['Partition'] == 0][target_attrs].reset_index(drop=True)
    val_df   = df[df['Partition'] == 1][target_attrs].reset_index(drop=True)
    test_df  = df[df['Partition'] == 2][target_attrs].reset_index(drop=True)

    train_dataset = ImageDatasetLoader(train_df, IMG_DIR, transform)
    val_dataset   = ImageDatasetLoader(val_df, IMG_DIR, transform)
    test_dataset  = ImageDatasetLoader(test_df, IMG_DIR, transform)

    # 5. Build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Num_TrainSample: {len(train_dataset)}, Num_ValSample: {len(val_dataset)}, Num_TestSample: {len(test_dataset)}")
    return train_loader, val_loader, test_loader