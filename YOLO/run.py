import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from ultralytics import YOLO  # YOLOv5 framework


# === Paths ===
current_directory = os.getcwd()
dataset_dir = os.path.join(current_directory, "dataset")
data_yaml = os.path.join(dataset_dir, "data.yaml")

# Debug: Print paths
print(f"Current Directory: {current_directory}")
print(f"Dataset Directory: {dataset_dir}")
print(f"Data YAML Path: {data_yaml}")

# Verify if paths exist
if not os.path.exists(dataset_dir):
    print(f"Error: Dataset directory does not exist at {dataset_dir}")
if not os.path.exists(data_yaml):
    print(f"Error: data.yaml file does not exist at {data_yaml}")

# Verify contents of dataset directory
print("Contents of dataset directory:")
print(os.listdir(dataset_dir))

# === Model Setup ===
def get_model():
    """Load the YOLO model."""
    model = YOLO('yolov5su.pt')  # Pretrained YOLOv5s model
    return model

model = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print("Model loaded successfully and is running on", device)

# === Training ===
def train_model(model, data_yaml, num_epochs=50, batch_size=8):
    """Train YOLOv5 on the specified dataset."""
    print("\n=== Starting Training ===")
    model.train(data=data_yaml, epochs=num_epochs, batch=batch_size, project="./runs")

# === Validation ===
def validate_model(model, data_yaml):
    """Validate YOLOv5 on the validation dataset."""
    print("\n=== Validating Model ===")
    results = model.val(data=data_yaml)
    print(f"Validation Results: {results}")
    return results

# === Run Training and Validation ===

train_model(model, data_yaml=data_yaml, num_epochs=50, batch_size=10)
validate_model(model, data_yaml=data_yaml)
