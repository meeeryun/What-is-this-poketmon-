import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os

# Definition of Models

# 1. AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 2. ResNet18 (Pre-trained)
def get_resnet18(num_classes):
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# 3. VGG16 (Pre-trained)
def get_vgg16(num_classes):
    model = models.vgg16(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model

# 4. MobileNetV2 (Pre-trained)
def get_mobilenet(num_classes):
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# Settings and Data load
st.set_page_config(page_title="Vision Experiment Dashboard", layout="wide")
st.title("🔬 Multi-Backbone Classifier Experiment")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def prepare_data(data_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_db, test_db = random_split(dataset, [train_size, test_size])
    return train_db, test_db, dataset.classes

# Choose models and parameter in streamlit
st.sidebar.header("⚙️ Settings")
backbone_name = st.sidebar.selectbox("Select Backbone", ["AlexNet", "ResNet18", "VGG16", "MobileNetV2"])
data_folder = st.sidebar.text_input("Dataset Folder Path", value="./mini_dataset")
epochs = st.sidebar.slider("Epochs", 1, 20, 5)
batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64], value=16)

if os.path.exists(data_folder):
    train_db, test_db, classes = prepare_data(data_folder)
    st.sidebar.success(f"Load OK! (Classes: {len(classes)})")
    
    # Reset Models
    if backbone_name == "AlexNet":
        model = AlexNet(num_classes=len(classes)).to(device)
    elif backbone_name == "ResNet18":
        model = get_resnet18(len(classes)).to(device)
    elif backbone_name == "VGG16":
        model = get_vgg16(len(classes)).to(device)
    else:
        model = get_mobilenet(len(classes)).to(device)
else:
    st.sidebar.error("Path Error!")
    st.stop()

# Main screen
tab1, tab2 = st.tabs(["Training", "Inference"])

with tab1:
    st.header(f"Training: {backbone_name}")
    if st.button("Start Training"):
        train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, drop_last=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        status = st.status(f"🚀 {backbone_name} Learning...", expanded=True)
        progress_bar = st.progress(0)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if i % 10 == 0:
                    print(f"[{backbone_name}] Ep {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(train_loader)
            status.update(label=f"✅ Epoch {epoch+1} Compelte (Loss: {avg_loss:.4f})", state="running")
            progress_bar.progress((epoch + 1) / epochs)
        
        save_name = f"{backbone_name.lower()}_model.pth"
        torch.save(model.state_dict(), save_name)
        status.update(label="Learning and Save complete", state="complete", expanded=False)
        st.success(f"Saved as {save_name}")
        
        with open(save_name, "rb") as f:
            st.download_button("Download Save file", f, file_name=save_name)

with tab2:
    st.header("Test Inference")
    uploaded_file = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, width=300)
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = test_transform(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            top5_p, top5_i = torch.topk(prob, min(5, len(classes)))
            for i in range(len(top5_p)):
                st.write(f"**{classes[top5_i[i]]}**: {top5_p[i].item()*100:.2f}%")