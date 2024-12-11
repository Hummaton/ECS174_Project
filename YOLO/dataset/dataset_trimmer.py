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
    './train/labels'
]

# Verify if directories exist
for input_dir in input_dirs:
    if not os.path.exists(input_dir):
        print(f"Error: Directory does not exist: {input_dir}")
    else:
        print(f"Directory exists: {input_dir}")

# Counters for each class
class_counts = {class_id: 0 for class_id in class_map.values()}

# Count instances of each class
for input_dir in input_dirs:
    if os.path.exists(input_dir):
        for label_file in os.listdir(input_dir):
            if label_file.endswith('.txt'):
                input_path = os.path.join(input_dir, label_file)
                with open(input_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # YOLO format: class_id x_center y_center width height
                            class_id = int(parts[0])
                            if class_id in class_map.values():
                                class_counts[class_id] += 1
                                print(f"Found class ID {class_id} in file {label_file}")

print("Initial class counts:", class_counts)

# Define the maximum allowed instances for mild, moderate, and severe corrosion
max_instances = {
    class_map['mild-corrosion']: 1200,
    class_map['moderate-corrosion']: 1200,
    class_map['severe-corrosion']: 1200
}

# Trim dataset
for input_dir in input_dirs:
    if os.path.exists(input_dir):
        for label_file in os.listdir(input_dir):
            if label_file.endswith('.txt'):
                input_path = os.path.join(input_dir, label_file)
                with open(input_path, 'r') as file:
                    lines = file.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # YOLO format: class_id x_center y_center width height
                        class_id = int(parts[0])
                        if class_id in max_instances and class_counts[class_id] > max_instances[class_id]:
                            class_counts[class_id] -= 1
                            continue
                        new_lines.append(line)

                if new_lines:
                    with open(input_path, 'w') as file:
                        file.write(''.join(new_lines))
                else:
                    os.remove(input_path)
                    image_path = input_path.replace('labels', 'images').replace('.txt', '.jpg')
                    if os.path.exists(image_path):
                        os.remove(image_path)

print("Final class counts:", class_counts)