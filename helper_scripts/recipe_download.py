import kagglehub
import pandas as pd
import os

# Download the dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("wilmerarltstrmberg/recipe-dataset-over-2m")
print(f"Dataset downloaded to: {path}")

# Find the CSV file in the downloaded folder
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the downloaded dataset.")

dataset_file_path = os.path.join(path, csv_files[0])
print(f"Loading {dataset_file_path}...")

# Load the dataset
df = pd.read_csv(dataset_file_path)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("First 5 records:")
print(df.head())

# Save to CSV in the output directory
os.makedirs("output", exist_ok=True)
save_path = "output/recipe_dataset_2m.csv"
df.to_csv(save_path, index=False)
print(f"\nSaved to: {os.path.abspath(save_path)}")
print(f"File size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")