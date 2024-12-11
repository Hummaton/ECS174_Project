import os

# Class mapping
class_map = {
    'Rust': 0,
    'car': 1,
    'copper-corrosion': 2,
    'corroded-part': 3,
    'corrosion': 4,
    'iron-rust': 5,
    'mild-corrosion': 6,
    'moderate-corrosion': 7,
    'rust': 8,
    'severe-corrosion': 9
}

# Paths
input_dirs = [
    './valid/labels', 
    './test/labels',  
    './train/labels'
]

image_width = 640  # Replace with your image width
image_height = 640  # Replace with your image height

for input_dir in input_dirs:
    for label_file in os.listdir(input_dir):
        if label_file.endswith('.txt'):
            input_path = os.path.join(input_dir, label_file)

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
                
                if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1 or width < 0 or width > 1 or height < 0 or height > 1:
                    print(f"Skipping out-of-bounds label in {label_file}: {line}")
                    continue

                # Append converted line
                converted_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            if converted_lines:
                # Overwrite the original label file
                with open(input_path, 'w') as file:
                    file.write('\n'.join(converted_lines))
            else:
                # Delete the file if no valid labels
                os.remove(input_path)

print(f"Converted labels to YOLO format")