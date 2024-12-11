import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from ultralytics import YOLO  # YOLOv5 framework
import matplotlib.pyplot as plt

# === Data Preprocessing ===
def convert_to_yolo_format(bbox_list, image_width, image_height):
    """Convert bounding boxes from [x_min, y_min, x_max, y_max] to YOLO format [x_center, y_center, width, height]."""
    yolo_bboxes = []
    for x_min, y_min, x_max, y_max in bbox_list:
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        yolo_bboxes.append([x_center, y_center, width, height])
    return yolo_bboxes

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
    model = YOLO('yolov5s.pt')  # Pretrained YOLOv5s model
    return model

model = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print("Model loaded successfully and is running on", device)

# === Training ===
def train_model(model, data_yaml, num_epochs=50, batch_size=8):
    """Train YOLOv5 on the specified dataset."""
    print("\n=== Starting Training ===")
    model.train(data=data_yaml, epochs=num_epochs, batch=batch_size)

# === Validation ===
def validate_model(model, data_yaml):
    """Validate YOLOv5 on the validation dataset."""
    print("\n=== Validating Model ===")
    results = model.val(data=data_yaml)
    print(f"Validation Results: {results}")
    return results

# === Testing ===
def test_model(model, image_dir):
    """Run YOLOv5 inference on test images."""
    print("\n=== Testing Model ===")
    results = model.predict(source=image_dir)
    for result in results:
        print(result)  # Prints bounding boxes, class labels, and confidence scores
    return results

# === Loss Plotting (Optional) ===
def plot_loss_curve(log_dir):
    """Plot training and validation loss curves."""
    loss_file = os.path.join(log_dir, "results.csv")
    if os.path.exists(loss_file):
        import pandas as pd
        data = pd.read_csv(loss_file)
        plt.plot(data["epoch"], data["train/box_loss"], label="Box Loss")
        plt.plot(data["epoch"], data["train/obj_loss"], label="Objectness Loss")
        plt.plot(data["epoch"], data["train/cls_loss"], label="Classification Loss")
        plt.plot(data["epoch"], data["val/box_loss"], label="Validation Box Loss", linestyle="--")
        plt.plot(data["epoch"], data["val/obj_loss"], label="Validation Objectness Loss", linestyle="--")
        plt.plot(data["epoch"], data["val/cls_loss"], label="Validation Classification Loss", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curves")
        plt.grid()
        plt.show()
    else:
        print("No loss file found for plotting.")

# === Run Training and Testing ===
train_model(model, data_yaml=data_yaml, num_epochs=10, batch_size=8)
validate_model(model, data_yaml=data_yaml)
test_model(model, image_dir=os.path.join(dataset_dir, "test/images"))