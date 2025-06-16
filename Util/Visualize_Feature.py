from Model import MobileNetV2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import re
import os
  

# Define transforms
transform = transforms.Compose([
    transforms.Resize((218, 178)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # transforms.Normalize([0.5]*3, [0.5]*3)
])


def get_latest_checkpoint(model_dir):
    # Match the checkpoint，eg. Epoch3_20250612_223344.pth
    pattern = r"Epoch(\d+)_\d{8}_\d{6}\.pth"
    new_epoch = -1
    new_path = None
    for filename in os.listdir(model_dir):
        match = re.match(pattern, filename)
        if filename == 'best_model.pth':
            new_epoch = None
            new_path = os.path.join(model_dir, filename)
            break
        elif match:
            epoch_num = int(match.group(1))
            if epoch_num > new_epoch:
                new_epoch = epoch_num
                new_path = os.path.join(model_dir, filename)
    print(f"Load the saved model: {new_path}, Epoch: {new_epoch}")
    return new_path, new_epoch


def visualize_feature_map(model, image_path, device):
    # Read image and transform
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model = model.to(device)
    # Get the feature of 4 layer
    _, (ori_feat, low_feat, mid_feat, high_feat) = model(image)

    feature_maps = [ori_feat[0], low_feat[0], mid_feat[0], high_feat[0]]  # shape: (C, H, W)
    titles = ['Ori-level', 'Low-level', 'Mid-level', 'High-level']

    plt.figure(figsize=(12, 9))

    for row in range(4):  # 4 chanel
        fmap = feature_maps[row].detach().cpu()
        for col in range(4):  # 4 channel every layer
            idx = row * 4 + col + 1
            plt.subplot(4, 4, idx)
            plt.imshow(fmap[col*3], cmap='viridis')  # or hot
            plt.title(f'{titles[row]}\nChannel {col*3}')
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_feature(feature_map, title='', num_cols=8):
    """
    feature_map: Tensor of shape [C, H, W]
    num_cols: Number of subplots per row
    """
    if feature_map.dim() == 4:
        feature_map = feature_map.squeeze(0)  # Remove batch dimension
    
    C, H, W = feature_map.shape
    num_cols = min(num_cols, C)
    num_rows = int(np.ceil(C / num_cols))

    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    for i in range(C):
        plt.subplot(num_rows, num_cols, i + 1)
        fmap = feature_map[i].detach().cpu().numpy()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)  # Normalize to [0,1]
        plt.imshow(fmap, cmap='viridis')
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # === Load the config  ===
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    MODEL_DIR = config['MODEL_DIR']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MobileNetV2(num_classes=10)
    image = "Dataset/img/000010.jpg"
    save_model, _ = get_latest_checkpoint(MODEL_DIR)
    model.load_state_dict(torch.load(save_model, map_location=device))

    visualize_feature_map(model, image, device)  # 同时可视化四层特征
    image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model = model.to(device)
    _, (ori_feat, low_feat, mid_feat, high_feat) = model(image)
    # visualize_feature(ori_feat, title='Ori Feature (Stem)')  # 32 channels
    # visualize_feature(low_feat, title='Low-Level Feature')  # 24 channels
    # visualize_feature(mid_feat, title='Mid-Level Feature')  # 64 channels
    # visualize_feature(high_feat, title='High-Level Feature')  # 180 channels