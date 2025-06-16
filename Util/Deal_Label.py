import yaml
import pandas as pd

# === Load the yaml config ===
with open("Config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Read the label file path
LABEL_PATH = config["LABEL_PATH"]

# Attribute to retain
target_attrs = ['Black_Hair', 'Eyeglasses', 'Heavy_Makeup', 'Male', 'Mustache',
                'Pale_Skin', 'Smiling', 'Wavy_Hair', 'Wearing_Hat', 'Young']

# === Read label file ===
with open(LABEL_PATH, 'r') as f:
    lines = f.readlines()

# Get the attribute name
attribute_names = lines[1].strip().split()

# Read dataset
data = []
for line in lines[2:]:
    parts = line.strip().split()
    filename = parts[0]
    labels = list(map(int, parts[1:]))
    data.append([filename] + labels)

# Create DataFrame
df = pd.DataFrame(data, columns=['Filename'] + attribute_names)

# Only retain the required columns
df = df[['Filename'] + target_attrs]

# Sort by Filename
df = df.sort_values(by='Filename').reset_index(drop=True)

# Save to txt file
output_path = "Dataset/list_attr.txt"
df.to_csv(output_path, sep=' ', index=False, header=True)

print(f"Saved to: {output_path}")
