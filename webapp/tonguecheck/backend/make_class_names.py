from torchvision import datasets
import json
from pathlib import Path

DATASET_DIR = r"C:\Users\sahal\OneDrive\Desktop\project\Mydata-2"

dataset = datasets.ImageFolder(DATASET_DIR)

class_names = dataset.classes

output_path = Path("class_names.json")

with open(output_path, "w") as f:
    json.dump(class_names, f, indent=2)

print("Class names saved successfully:")
print(class_names)