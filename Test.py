from Util.Load_Dataset import Load_Dataset
from Util.Model import MobileNetV2
from torch.nn import BCEWithLogitsLoss  # Multi-label regression
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse
import torch
import yaml
import re
import os

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
    return new_path, new_epoch

def test_model(model, test_dataloader, device, model_dir):
    model = model.to(device)
    criterion = BCEWithLogitsLoss()

    save_model_path, save_epoch = get_latest_checkpoint(model_dir)
    if save_model_path:
        print(f"Resume From Saved: {save_model_path} (Epoch: {save_epoch})")
        model.load_state_dict(torch.load(save_model_path))
    else:
        print("No recoverable model was found!")

    # === Test ===
    model.eval()  # 关闭 Dropout / BatchNorm 等训练特性
    correct_val = total_val = 0
    testing_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            outputs, _, = model(images)

            loss = criterion(outputs, labels)
            testing_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct_val += (preds == labels).sum().item()
            total_val += labels.numel()

        test_loss = testing_loss / len(test_dataloader)
        test_acc = correct_val / total_val
        print("Predict Label:", preds[0].cpu().numpy())
        print("Real Label   :", labels[0].cpu().numpy())
        print(f"Test Loss  : {test_loss:.4f}, Test Acc  : {test_acc:.4f}")

    # === Save log ===
    with open("Log/train.log", 'a') as f:
        f.write(
            f"Testing, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n\n"
        )


def predict_single_image(model, image_path, device, model_dir):
    save_model_path, save_epoch = get_latest_checkpoint(model_dir)
    if save_model_path:
        print(f"Load Model From Saved: {save_model_path} (Epoch: {save_epoch})")
        model.load_state_dict(torch.load(save_model_path, map_location=device))
    else:
        print("No recoverable model was found!")

    model.eval()
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = model(input_tensor)
        preds = (torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy()

    # === Load real label from list_attr.txt ===
    import pandas as pd
    img_name = os.path.basename(image_path)
    attr_df = pd.read_csv(LABEL_PATH, sep=r'\s+', skiprows=1, index_col=0)
    if img_name in attr_df.index:
        true_label = attr_df.loc[img_name].values.astype(int)
    else:
        print(f"⚠️ Warning: {img_name} not found in label file.")
        true_label = None

    # === Result ===
    print(f"\n🔍 Prediction for: {image_path}")
    label_map = config.get('ATTRIBUTES', [f"Attr{i}" for i in range(1, 11)])
    for label, pred in zip(label_map, preds):
        print(f"{label:15}: {'✅' if pred else '❌'}")

    if true_label is not None:
        print(f"\n📌 Ground Truth:")
        for label, true in zip(label_map, true_label):
            print(f"{label:15}: {'✅' if true else '❌'}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='Path to a single image for prediction')
    args = parser.parse_args()
    # === Load the config  ===
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    IMG_DIR = config['IMG_DIR']
    MODEL_DIR = config['MODEL_DIR']
    LABEL_PATH = config['LABEL_PATH']
    PARTITION_PATH = config['PARTITION_PATH']
    IMAGE_SIZE = config['IMAGE_SIZE']
    USE_CUDA = config['USE_CUDA']

    # === Define Model ===
    if USE_CUDA and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda is avilable, Running on Cuda")
    elif USE_CUDA:
        device = torch.device("cpu")
        print("Cuda is unavilable, Running on Cpu")
    else:
        device = torch.device("cpu")
        print("Select to Running on Cpu")

    model = MobileNetV2(num_classes=10)

    if args.img:
        predict_single_image(model.to(device), args.img, device, MODEL_DIR)
    else:
        _, _, test_dataloader = Load_Dataset(LABEL_PATH, IMG_DIR, PARTITION_PATH)
        test_model(model, test_dataloader, device, MODEL_DIR)



