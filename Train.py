from Util.Load_Dataset import Load_Dataset
from Util.Model import MobileNetV2
from torch.nn import BCEWithLogitsLoss  # Multi-label regression
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch
import torch.optim as optim
import time
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
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > new_epoch:
                new_epoch = epoch_num
                new_path = os.path.join(model_dir, filename)
    return new_path, new_epoch


def train_model(model, train_dataloader, val_dataloader, device, total_epoch, save_frequency, model_dir, resume_train):
    model = model.to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # factor 控制学习率缩小的幅度（如 0.5 是缩小一半）
    # patience 是“忍耐期”，即在 patience 个 epoch 内val_loss 没有下降才会调整学习率
    # verbose=True 会自动打印 learning rate 变化日志
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    if SAVE_TO_TENSORBOARD:
        # Initialize the Tensorboard
        # log_dir = os.path.join("Log/Board", time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        log_dir = "Log/Board/MobileNet-1000"
        writer = SummaryWriter(log_dir)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.makedirs(model_dir, exist_ok=True)  # Make sure the dir exist

    # === If Load Saved Model ===
    if resume_train:
        save_model_path, start_epoch = get_latest_checkpoint(model_dir)
        if save_model_path:
            print(f"Resume From Saved: {save_model_path} (From Epoch {start_epoch + 1} to Epoch {total_epoch + 1})")
            model.load_state_dict(torch.load(save_model_path))
        else:
            print("No recoverable model was found and training will start from Epoch 1")
            start_epoch = 0
    else:
        start_epoch = 0

    for epoch in range(start_epoch, total_epoch):
        print(f"✅ Epoch [{epoch+1}/{total_epoch}]")

        # === Begin the train ===
        model.train()
        training_loss = 0.0
        correct_train = total_train = 0

        for images, labels in tqdm(train_dataloader, desc=f"Training"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # outputs = model(images)
            outputs, _, = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct_train += (preds == labels).sum().item()
            total_train += labels.numel()

        train_loss = training_loss / len(train_dataloader)
        train_acc = correct_train / total_train
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        if SAVE_TO_TENSORBOARD:
            # Write training metrics to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('Accuracy/train', train_acc, epoch + 1)


        # === Evaluation ===
        model.eval()  # 关闭 Dropout / BatchNorm 等训练特性
        correct_val = total_val = 0
        valing_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc="Evaluating"):
                images = images.to(device)
                labels = labels.to(device)
                # outputs = model(images)
                outputs, _, = model(images)

                loss = criterion(outputs, labels)
                valing_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                correct_val += (preds == labels).sum().item()
                total_val += labels.numel()

            val_loss = valing_loss / len(val_dataloader)
            val_acc = correct_val / total_val
            # print("Predict Label:", preds[0].cpu().numpy())
            # print("Real Label   :", labels[0].cpu().numpy())
            print(f"Val Loss  : {val_loss:.4f}, Val Acc  : {val_acc:.4f}")

            # scheduler.step(val_loss)

        if SAVE_TO_TENSORBOARD:
            # Write validation metrics to TensorBoard
            writer.add_scalar('Loss/val', val_loss, epoch + 1)
            writer.add_scalar('Accuracy/val', val_acc, epoch + 1)

        # === Early stopp ===
        best_loss = 1
        patience = 5  # Tolerate count
        counter = 0
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            if SAVE_BEST_MODEL:
                torch.save(model.state_dict(), os.path.join(model_dir, f"best_model.pth"))  # Save the best model
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered. The best loss is {best_loss:.4f}")
                break


        # === Save model ===
        if_save = ""
        if SAVE_MODEL and (epoch + 1) % save_frequency == 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            save_path = os.path.join(model_dir, f"Epoch{epoch+1}_{timestamp}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"The model has been saved to: {save_path}")
            if_save = "(saved)"


        # === Save log ===
        if SAVE_LOG:
            current_lr = optimizer.param_groups[0]['lr']
            save_log(epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr, if_save, "Log/train.log")
        
        if SAVE_TO_TENSORBOARD:
            # Close the Tensorboard Writer
            writer.close()

    with open("Log/train.log", 'a') as f:
        f.write(f"\n")


# Save Log As File
def save_log(epoch, train_loss, train_acc, val_loss, val_acc, lr, if_save, log_file):
    with open(log_file, 'a') as f:
        f.write(
            f"Epoch {epoch}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"lr: {lr} {if_save}\n"
        )


if __name__ == '__main__':
    # === Load the config  ===
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    IMG_DIR = config['IMG_DIR']
    MODEL_DIR = config['MODEL_DIR']
    LABEL_PATH = config['LABEL_PATH']
    PARTITION_PATH = config['PARTITION_PATH']
    LOG_PATH = config['LOG_PATH']
    TOTAL_EPOCH = config['TOTAL_EPOCH']
    SAVE_FREQUENCE = config['SAVE_FREQUENCE']
    LEARNING_RATE = config['LEARNING_RATE']
    SAVE_MODEL = config['SAVE_MODEL']
    SAVE_BEST_MODEL  = config['SAVE_BEST_MODEL']
    SAVE_TO_TENSORBOARD = config['SAVE_TO_TENSORBOARD']
    SAVE_LOG = config['SAVE_LOG']
    RESUME_TRAIN = config['RESUME_TRAIN']
    USE_CUDA = config['USE_CUDA']

    # === Load Dataset ===
    train_dataloader, val_dataloader, test_dataloader = Load_Dataset(LABEL_PATH, IMG_DIR, PARTITION_PATH)

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

    # === Start Train ===
    train_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        device, 
        TOTAL_EPOCH, 
        SAVE_FREQUENCE, 
        MODEL_DIR,
        RESUME_TRAIN
        )