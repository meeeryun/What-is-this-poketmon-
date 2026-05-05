# What-is-this-poketmon-

## Image Classification: Multi-Backbone Benchmarking
This project evaluates and compares the performance of various CNN architectures—ranging from the classic AlexNet to modern backbones like ResNet, VGG, and MobileNet—on a challenging dataset with 150 categories.

### 1. Experimental Setup
-Dataset: 150 Classes (Mini-dataset: 20 images per class, 3,000 images total)
(원래는 7000+ 데이터셋이었지만 일반 노트북 CPU로는 감당이 안 되는 양이라 Batch를 제일 낮게 해서 돌려도 Epoch 학습 중 끊기는 현상이 다수 발생해서 개수를 조정했습니다.)

-Device: CPU-based training environment

-Frameworks: PyTorch, Torchvision, Streamlit

-Optimization: Adam Optimizer (LR: 0.001), Batch Size: 16

### 2. Architectures & Methodology
1. AlexNet (Training from Scratch): A landmark 8-layer architecture. We implemented this model manually to understand the fundamental mechanics of CNNs and observed its behavior when trained without prior weights.
2. ResNet18 (Transfer Learning): Utilizes "Skip Connections" to mitigate the vanishing gradient problem. Pre-trained on ImageNet for superior feature extraction.
3. VGG16 (Transfer Learning): Employs a deep stack of $3 \times 3$ convolutions, providing high precision in feature localization.
4. MobileNetV2 (Transfer Learning): Optimized for mobile and CPU efficiency using Depthwise Separable Convolutions.

### 3. Comparative Analysis (AlexNet Results)

| Model Name | Training Mode | Final Loss | 
| :--- | :---: | :---: |
| **AlexNet** | Scratch  | 5.1221 | 
| **ResNet18** | Pre-trained  | 1.8240 | 
| **VGG16** | Pre-trained | 2.1055 | 
| **MobileNetV2** | Pre-trained | 1.9580 |

### "Unexpected Performance of ResNet18"

Contrary to expectations, ResNet18 showed lower-than-anticipated accuracy during the initial epochs.

This could be attributed to a 'Domain Gap' between the pre-trained ImageNet weights and the specific features of our 150-class dataset.

Furthermore, the small number of samples per class (20 images) might have hindered the model's ability to fine-tune its deep residual layers effectively within a limited time frame.
<img width="808" height="1062" alt="image" src="https://github.com/user-attachments/assets/59e1d664-4b67-44c6-bfae-84902883d081" />


이번 과제도 성능 자체로 힘듦이 있어 최대한 데이터셋을 줄이고 줄여 150 클래스 X 20장 써서 3000 클래스로 완성하여 작동 시켜봤습니다.




