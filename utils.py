import os

def check_dataset_structure(path="data"):
    for folder in os.listdir(path):
        print(f"{folder}: {len(os.listdir(os.path.join(path, folder)))} images")