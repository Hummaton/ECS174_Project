import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import numpy as np
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

def parse_yolo_labels(label_file):
    """Parse YOLO label files and return ground-truth labels and bounding boxes."""
    labels, bboxes = [], []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # Skip invalid lines
            labels.append(int(parts[0]))  # Class ID
            bboxes.append([float(x) for x in parts[1:]])  # Normalized bbox [x_center, y_center, width, height]
    return labels, bboxes

def calculate_metrics(y_true, y_pred, y_pred_scores, iou_threshold=0.5):
    """Calculate precision, recall, accuracy, and F1-score."""
    # Binary classification for each prediction
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n=== Evaluation Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return precision, recall, f1, accuracy

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
def test_model(model, image_dir, label_dir):
    """Run YOLOv5 inference on test images and calculate metrics."""
    print("\n=== Testing Model ===")
    results = model.predict(source=image_dir, save=False)

    all_true_labels = []
    all_pred_labels = []

    # Iterate over predictions and compare with ground-truth labels
    for result in results:
        image_path = result.path
        image_name = os.path.basename(image_path).rsplit('.', 1)[0]
        label_file = os.path.join(label_dir, f"{image_name}.txt")
        
        if not os.path.exists(label_file):
            print(f"Warning: No label file found for {image_name}. Skipping...")
            continue

        # Parse ground-truth labels
        true_labels, _ = parse_yolo_labels(label_file)
        all_true_labels.extend(true_labels)

        # Get YOLO predictions
        if result.boxes is not None:  # Ensure there are predictions
            pred_labels = result.boxes.cls.cpu().numpy().astype(int).tolist()
        else:
            pred_labels = []  # No predictions for this image
        
        # Append predictions
        all_pred_labels.extend(pred_labels)

        # Debug: Print predictions and ground truth for this image
        print(f"Image: {image_name}")
        print(f"Ground Truth: {true_labels}")
        print(f"Predictions: {pred_labels}")

    # Ensure consistent lengths
    min_length = min(len(all_true_labels), len(all_pred_labels))
    all_true_labels = all_true_labels[:min_length]
    all_pred_labels = all_pred_labels[:min_length]

    # Calculate metrics
    precision, recall, f1, accuracy = calculate_metrics(
        y_true=all_true_labels,
        y_pred=all_pred_labels,
        y_pred_scores=None  # Scores not used in this version
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "results": results
    }

# === Loss Plotting (Optional) ===
def plot_loss_curve(log_dir):
    """Plot training and validation loss curves."""
    loss_file = os.path.join(log_dir, "results.csv")
    if os.path.exists(loss_file):
        import pandas as pd
        data = pd.read_csv(loss_file)
        plt.plot(data["epoch"], data["train/box_loss"], label="Box Loss")
        plt.plot(data["epoch"], data["train/dfl_loss"], label="DFL Loss")
        plt.plot(data["epoch"], data["train/cls_loss"], label="Classification Loss")

        plt.plot(data["epoch"], data["val/box_loss"], label="Validation Box Loss", linestyle="--")
        plt.plot(data["epoch"], data["val/dfl_loss"], label="Validation DFL Loss", linestyle="--")
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



train_model(model, data_yaml=data_yaml, num_epochs=15, batch_size=8)