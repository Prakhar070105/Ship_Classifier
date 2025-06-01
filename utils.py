import os
import pandas as pd

def load_all_labels(root_dir):
    data = []

    for ship_class in os.listdir(root_dir):
        class_path = os.path.join(root_dir, ship_class)
        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):
            if not file_name.endswith(".wav"):
                continue

            # Build path to labels.csv in same folder
            ship_name = os.path.splitext(file_name)[0]  # remove .wav
            meta_path = os.path.join(class_path, "labels.csv")  # assume labels.csv is in class folder

            if os.path.exists(meta_path):
                try:
                    df = pd.read_csv(meta_path)
                    df["filepath"] = os.path.join(class_path, file_name)
                    df["ship_type"] = ship_class
                    data.append(df)
                except Exception as e:
                    print(f"Error reading {meta_path}: {e}")
            else:
                print(f"labels.csv not found at: {meta_path}")

    if not data:
        raise FileNotFoundError("No labels.csv files were loaded. Check your directory structure.")

    return pd.concat(data, ignore_index=True)

