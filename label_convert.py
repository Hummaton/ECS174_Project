import os

# Class mapping
class_map = {
    'Rust': 0,
    'car': 1,
    'copper corrosion': 2,
    'corroded-part': 3,
    'corrosion': 4,
    'iron rust': 5,
    'mild-corrosion': 6,
    'moderate-corrosion': 7,
    'rust': 8,
    'severe-corrosion': 9
}

# Paths
input_dir = 'yolov5/dataset/train/labels'  # Update to your labels directory
output_dir = 'yolov5/dataset/train/labels_YOLO'  # Directory to save converted labels
image_width = 640  # Replace with your image width
image_height = 640  # Replace with your image height

os.makedirs(output_dir, exist_ok=True)

for label_file in os.listdir(input_dir):
    if label_file.endswith('.txt'):
        input_path = os.path.join(input_dir, label_file)
        output_path = os.path.join(output_dir, label_file)

        with open(input_path, 'r') as file:
            lines = file.readlines()

        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                print(f"Skipping invalid line in {label_file}: {line}")
                continue
            
            # Extract bounding box coordinates and class name
            x_min, y_min = float(parts[0]), float(parts[1])
            x_max, y_max = float(parts[4]), float(parts[5])
            class_name = parts[8]
            
            # Convert to YOLO format
            if class_name not in class_map:
                print(f"Unknown class name '{class_name}' in {label_file}")
                continue
            
            class_id = class_map[class_name]
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            # Append converted line
            converted_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Write to new label file
        with open(output_path, 'w') as file:
            file.write('\n'.join(converted_lines))