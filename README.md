# Facial_Attributes_Recognization_Based_On_MobileNetV2
A face multi-attribute classification method based on MobileNetV2
## 🧠Overview
This project implements a lightweight, multi-label facial attribute recognition system leveraging the MobileNetV2 architecture. It is capable of classifying multiple attributes (e.g., **gender, expression, pose**, etc.) from facial images with high efficiency. The system is optimized for training and inference on devices with limited computational resources.

The complete pipeline includes **data preprocessing, model definition, training, evaluation, and visualization of results**.
## 🔁Workflow
```
graph TD
    A[Original Dataset] --> B[Data Preprocessing & Partition]
    B --> C[Model Construction (MobileNetV2)]
    C --> D[Model Training]
    D --> E[Evaluation & Testing]
    E --> F[Attribute Prediction on New Images]
    D --> G[Visualization with TensorBoard]
```


## 📦Prerequisites
Download the project:
```
git clone https://github.com/Wanderer946/Facial_Attributes_Recognization_Based_On_MobileNetV2.git
cd Facial_Attributes_Recognization_Based_On_MobileNetV2
```
### 🛠Dependencies
- In conda:
Create the environment with:
```bash
conda env create -f environment.yaml
conda activate face_attr
```
- In pip:
Install dependencies with:
```bash
pip install -r requirements.txt
```
**Key dependencies:**
PyTorch
torchvision
pandas
numpy
Pillow
tqdm
pyyaml
matplotlib

### 📁Dataset
Before starting the train and test, you should decompress the compressed package of the data set in advance:
```bash
unzip Dataset/img/img.zip Dataset/img
```
Your dataset directory should follow this structure:
```bash
Dataset/
├─ img/
│  ├─ 000001.jpg
│  ├─ 000002.jpg
│  └─ ...
├─ list_attr.txt            # Attribute labels
├─ list_partition_1000.txt  # Partition files
├─ list_partition_2000.txt
├─ list_partition_3000.txt
└─ readme.md
```

## 🗂Project Structure

```
Project_Root/
├─Dataset                        # 数据集目录
│  └─img                         # 原始图像文件目录
│      ├─000001.jpg              # 图像样本
│      ├─000002.jpg
│      └─...                     # 更多图像样本
│  ├─list_attr.txt               # 所有图像对应的属性标签列表（多标签）
│  ├─list_partition_1000.txt     # 划分出的前1000张图像的训练/验证/测试索引
│  ├─list_partition_2000.txt     # 前2000张图像的划分索引
│  ├─list_partition_3000.txt     # 前3000张图像的划分索引
│  └─readme.md                   # 数据集说明文档
├─Log                            # 日志文件夹（训练过程记录与TensorBoard日志）
│  └─Board                       # TensorBoard 可视化日志
│      ├─MobileNet-1000-0.25    # 1000图像α=0.25训练的TensorBoard记录
│      ├─MobileNet-1000-0.50    # 1000图像α=0.50训练的TensorBoard记录
│      ├─MobileNet-1000-1.00    # 1000图像α=1.00训练的TensorBoard记录
│      ├─MobileNet-2000          # 2000图像α=1.00训练的TensorBoard记录
│      ├─MobileNet-3000
│      └─MobileNet-3000-lr-damp # 降学习率版本的训练记录
│  ├─console.log                 # 训练期间的控制台输出日志
│  └─train.log                   # 每轮训练与验证结果的日志
├─Model                          # 模型保存目录
│  └─MobileNet                   # MobileNet模型权重存储
│      ├─best_model.pth          # 表现最优的模型参数
│      ├─Epoch5_20250616_190701.pth  # 第5轮保存的模型（时间戳命名）
│      ├─Epoch10_20250616_190932.pth
│      └─...                     # 更多训练过程中保存的模型
├─Util                           # 工具脚本目录
│  ├─Deal_Label.py               # 标签文件的处理（清洗、格式转换）
│  ├─Load_Dataset.py             # 加载数据集与划分数据
│  ├─Model.py                    # 网络结构定义（MobileNetV2 及辅助模块）
│  ├─New_Partition.py            # 生成新的数据划分列表
│  └─Visualize_Feature.py        # 特征图的可视化方法
├─.env                           # 环境变量配置（用于本地开发）
├─Config.yaml                    # 项目参数配置文件（如文件路径、学习率等）
├─readme.md                      # 项目说明文档（功能介绍、运行方式等）
├─requirements.txt              # Python依赖库清单
├─Test.py                        # 模型测试与评估脚本
└─Train.py                       # 主训练程序
```

## 🚀Usage
### 1.Prepare Data
Ensure your dataset is organized as described above. Update the file paths in config.yaml.
### 2.Train the Model
```bash
python Train.py
```
Training logs and TensorBoard summaries will be saved in Log/.
### 3.Visualize Training Metrics
Start TensorBoard:
```bash
tensorboard --logdir Log/Board
```
Open `http://localhost:6006` in your browser to view training curves.
### 4.Test the Model
- Run testing on the test set:
```bash
python Test.py
```
- Or predict a single image's attributes:
```bash
python Test.py --img Dataset/img/000010.jpg
```
## ⚙Configuration
The config.yaml file contains key configuration parameters:
```yaml
# Dir
IMG_DIR: Dataset/img
MODEL_DIR: Model/MobileNet

# Path
LABEL_PATH: Dataset/list_attr.txt
PARTITION_PATH: Dataset/list_partition_3000.txt
LOG_PATH: Log/train.log

# Attributes（黑发、眼镜、浓妆、男性、胡子、白皮肤、微笑、卷发、帽子、年轻人）（1为active，-1为negative）
ATTRIBUTES: ['Black_Hair', 'Eyeglasses', 'Heavy_Makeup', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Wavy_Hair', 'Wearing_Hat', 'Young']

BATCH_SIZE: 32
IMAGE_SIZE: [218, 178]
NUM_WORKERS: 3

TOTAL_EPOCH: 50
SAVE_FREQUENCE: 5
LEARNING_RATE: 0.001

SAVE_MODEL: False
SAVE_BEST_MODEL: False
SAVE_TO_TENSORBOARD: True
SAVE_LOG: True
RESUME_TRAIN: False
USE_CUDA: True
```

## 📈Result
After training, the system can output prediction vectors representing each attribute with high accuracy. Sample output:
- **Training**:
```bash
     Filename  Black_Hair  Eyeglasses  Heavy_Makeup  Male  Mustache  Pale_Skin  Smiling  Wavy_Hair  Wearing_Hat  Young
0  000001.jpg           0           0             1     0         0          0        1          0            0      1
1  000002.jpg           0           0             0     0         0          0        1          0            0      1
2  000003.jpg           0           0             0     1         0          0        0          1            0      1
3  000004.jpg           0           0             0     0         0          0        0          0            0      1
4  000005.jpg           0           0             1     0         0          0        0          0            0      1
Num_TrainSample: 1600, Num_ValSample: 200, Num_TestSample: 200
Cuda is avilable, Running on Cuda
✅ Epoch [1/50]
Training: 100%|█████████████████████████████████████████████████████████████████████████████| 50/50 [00:16<00:00,  3.03it/s] 
Train Loss: 0.4714, Train Acc: 0.7634
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████| 7/7 [00:13<00:00,  1.92s/it] 
Val Loss  : 0.4740, Val Acc  : 0.7690
✅ Epoch [2/50]
Training: 100%|█████████████████████████████████████████████████████████████████████████████| 50/50 [00:15<00:00,  3.23it/s] 
Train Loss: 0.4285, Train Acc: 0.7943
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████| 7/7 [00:13<00:00,  1.86s/it] 
Val Loss  : 0.4852, Val Acc  : 0.7965
...
```

- **Testing**:
```bash
Cuda is avilable, Running on Cuda
     Filename  Black_Hair  Eyeglasses  Heavy_Makeup  Male  Mustache  Pale_Skin  Smiling  Wavy_Hair  Wearing_Hat  Young
0  000001.jpg           0           0             1     0         0          0        1          0            0      1
1  000002.jpg           0           0             0     0         0          0        1          0            0      1
2  000003.jpg           0           0             0     1         0          0        0          1            0      1
3  000004.jpg           0           0             0     0         0          0        0          0            0      1
4  000005.jpg           0           0             1     0         0          0        0          0            0      1
Num_TrainSample: 2400, Num_ValSample: 300, Num_TestSample: 300
Resume From Saved: Model/MobileNet\best_model.pth (Epoch: None)
Testing: 100%|█████████████████████████████████████████████████████████████████████████████| 10/10 [00:12<00:00,  1.28s/it]
Predict Label: [False False False  True False False  True False False False]
Real Label   : [0. 0. 0. 1. 0. 0. 1. 0. 0. 1.]
Test Loss  : 0.2952, Test Acc  : 0.9033
```
You can view the training effect intuitively on Tensorboard:
**Change value of alpha:**
![image](https://github.com/user-attachments/assets/401772fb-0729-418c-a9ba-dc4da78f8787)

**Change the number of img:**
![image](https://github.com/user-attachments/assets/0bffbbf9-2863-41a2-8da3-e8903d3d44a7)

**Viridis effect of different layer:**
![4e9431ef6ce0f4f4aecf76931b4b53c1](https://github.com/user-attachments/assets/ebacb931-79a3-4027-ae4a-d440d4dfe162)


## 🙏Acknowledgments
- The implementation is based on PyTorch.
- MobileNetV2 was introduced in MobileNetV2: Inverted Residuals and Linear Bottlenecks.
- Dataset is from CelebA.
