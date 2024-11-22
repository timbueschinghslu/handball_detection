import os
import shutil
import yaml

def read_classes_from_yaml(data_yaml_path):
    """Read class names from a data.yaml file."""
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    classes = data_yaml.get('names', [])
    return classes

def write_combined_data_yaml(combined_classes, output_dataset_path):
    """Write combined data.yaml to the output dataset."""
    data_yaml = {
        'train': os.path.join(output_dataset_path, 'train', 'images'),
        'val': os.path.join(output_dataset_path, 'valid', 'images'),
        'test': os.path.join(output_dataset_path, 'test', 'images'),
        'nc': len(combined_classes),
        'names': combined_classes
    }
    output_yaml_path = os.path.join(output_dataset_path, 'data.yaml')
    with open(output_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

def combine_labels(old_label, label_mapping):
    """Map old labels to new labels based on the label_mapping dictionary."""
    return label_mapping.get(old_label, old_label)  # Default to old label if not in mapping

def is_valid_label_line(line, num_classes):
    """Check if a label line is valid."""
    elements = line.strip().split()
    if len(elements) < 5:
        return False, "Label line does not have at least 5 elements."
    try:
        class_id = int(elements[0])
        if not (0 <= class_id < num_classes):
            return False, f"Class ID {class_id} is out of valid range (0 to {num_classes - 1})."
    except ValueError:
        return False, "Class ID is not an integer."
    # Additional checks for bounding box coordinates can be added here if needed
    return True, ""

def combine_datasets(dataset_paths, output_dataset_path):
    """
    Combine multiple YOLO-format datasets into one, with sanity checks.

    Args:
        dataset_paths (list of str): Paths to the datasets to combine.
        output_dataset_path (str): Path to the output combined dataset.
    """
    # Define label mapping as per your instructions
    label_mapping = {
        'ball': 'ball',
        'Ball': 'ball',
        'handball': 'ball',
        'goalkeeper': 'goalkeeper',
        'handballgoalkeeper': 'goalkeeper',
        'players': 'players',
        'player': 'players',
        'handballplayer': 'players',
        'Player': 'players',
        'referees': 'referees',
        'referee': 'referees',
        'handballreferee': 'referees',
        'referi': 'referees'
        # Other labels remain as they are
    }

    # Step 1: Build combined class list and class mappings
    combined_classes = []
    class_mappings = []  # List of dicts for mapping old class IDs to new IDs per dataset

    for dataset_idx, dataset_path in enumerate(dataset_paths):
        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
            dataset_classes = data_yaml.get('names', [])
        else:
            # If data.yaml does not exist, we can't proceed
            raise FileNotFoundError(f"data.yaml file not found in dataset: {dataset_path}")

        mapping = {}
        for idx, class_name in enumerate(dataset_classes):
            # Map old label to new label
            new_label = combine_labels(class_name, label_mapping)
            if new_label not in combined_classes:
                combined_classes.append(new_label)
            new_class_id = combined_classes.index(new_label)
            mapping[idx] = new_class_id
        class_mappings.append(mapping)

    num_classes = len(combined_classes)

    # Create output directories
    splits = ['train', 'valid', 'test']
    for split in splits:
        images_output_dir = os.path.join(output_dataset_path, split, 'images')
        labels_output_dir = os.path.join(output_dataset_path, split, 'labels')
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)

    # Write combined data.yaml to output dataset
    write_combined_data_yaml(combined_classes, output_dataset_path)

    # Initialize counters for summary
    summary = {dataset_path: {split: {'images': 0, 'labels': 0, 'missing_labels': 0, 'invalid_labels': 0} for split in splits} for dataset_path in dataset_paths}

    # Step 2: Process and copy images and labels for each split
    for dataset_idx, dataset_path in enumerate(dataset_paths):
        mapping = class_mappings[dataset_idx]

        for split in splits:
            # Default directories
            images_dir = os.path.join(dataset_path, split, 'images')
            labels_dir = os.path.join(dataset_path, split, 'labels')

            # Check if directories exist
            if not os.path.exists(images_dir):
                print(f"No images directory found for split '{split}' in dataset: {dataset_path}")
                continue
            if not os.path.exists(labels_dir):
                print(f"No labels directory found for split '{split}' in dataset: {dataset_path}")
                continue

            images_output_dir = os.path.join(output_dataset_path, split, 'images')
            labels_output_dir = os.path.join(output_dataset_path, split, 'labels')

            image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
            for img_file in image_files:
                src_img_path = os.path.join(images_dir, img_file)
                # Corresponding label file
                label_filename = os.path.splitext(img_file)[0] + '.txt'
                src_label_path = os.path.join(labels_dir, label_filename)

                if not os.path.exists(src_label_path):
                    print(f"Warning: No label file found for image '{img_file}' in dataset '{dataset_path}', split '{split}'. Skipping this image.")
                    summary[dataset_path][split]['missing_labels'] += 1
                    continue

                # Sanity check for label file
                with open(src_label_path, 'r') as f:
                    label_lines = f.readlines()

                valid_label_lines = []
                for line_num, line in enumerate(label_lines, start=1):
                    is_valid, error_message = is_valid_label_line(line, num_classes)
                    if not is_valid:
                        print(f"Warning: Invalid label in '{src_label_path}' at line {line_num}: {error_message}")
                        summary[dataset_path][split]['invalid_labels'] += 1
                        continue  # Skip invalid label line

                    # Process label line
                    elements = line.strip().split()
                    old_class_id = int(elements[0])
                    if old_class_id in mapping:
                        new_class_id = mapping[old_class_id]
                        # Replace old class ID with new class ID
                        new_line = ' '.join([str(new_class_id)] + elements[1:])
                        valid_label_lines.append(new_line)
                    else:
                        print(f"Warning: Class ID {old_class_id} not in mapping for dataset '{dataset_path}'. Skipping this label.")
                        summary[dataset_path][split]['invalid_labels'] += 1

                if not valid_label_lines:
                    print(f"Warning: No valid labels found for image '{img_file}' in dataset '{dataset_path}', split '{split}'. Skipping this image.")
                    summary[dataset_path][split]['invalid_labels'] += 1
                    continue

                # Copy image file
                dst_img_filename = f"{dataset_idx}_{split}_{img_file}"
                dst_img_path = os.path.join(images_output_dir, dst_img_filename)
                shutil.copy(src_img_path, dst_img_path)
                summary[dataset_path][split]['images'] += 1

                # Write new label file
                dst_label_filename = os.path.splitext(dst_img_filename)[0] + '.txt'
                dst_label_path = os.path.join(labels_output_dir, dst_label_filename)
                with open(dst_label_path, 'w') as f:
                    f.write('\n'.join(valid_label_lines))
                summary[dataset_path][split]['labels'] += 1

    # Print summary of file counts for each dataset and split
    print("\nSummary of files in each dataset and split:")
    for dataset_path in dataset_paths:
        print(f"\nDataset: {dataset_path}")
        for split in splits:
            images_count = summary[dataset_path][split]['images']
            labels_count = summary[dataset_path][split]['labels']
            missing_labels = summary[dataset_path][split]['missing_labels']
            invalid_labels = summary[dataset_path][split]['invalid_labels']
            print(f"  Split '{split}':")
            print(f"    Images processed: {images_count}")
            print(f"    Labels processed: {labels_count}")
            if missing_labels > 0:
                print(f"    Images skipped due to missing labels: {missing_labels}")
            if invalid_labels > 0:
                print(f"    Invalid labels skipped: {invalid_labels}")

    print(f"\nDatasets combined successfully into '{output_dataset_path}'")

if __name__ == '__main__':
    # List of dataset paths provided directly in the script
    dataset_paths = [
        'training/Handball-Detection-2',
        'training/handball-detection-6',
        'training/Handball-match-3',
        'training/maturarbeit-handballplayer-detc-3',
        'training/Test02-10',
        'training/Yolov8v2-1'
    ]

    # Output path for the combined dataset
    output_dataset_path = 'combined_dataset'

    # Combine the datasets with sanity checks
    combine_datasets(dataset_paths, output_dataset_path)
