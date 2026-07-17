import os

# Define your rename mapping
rename_map = {
    "B.npy": "background.npy",
    "C.npy": "names.npy",
    "I.npy": "wanted_z_and_mu.npy",
    "M.npy": "bin_edges.npy",
    "S.npy": "signal_shape.npy",
    "X.npy": "bin_content.npy",
    "Z.npy": "true_z.npy"
}

# Path to the root directory containing the folders
root_dir = "./temp/"  # Change this to the path where your folders are located

# Walk through each folder in the root directory
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)

    if os.path.isdir(folder_path) and folder_name.startswith("Valid_"): # Choose folders starting with "Valid_"
        for short_name, long_name in rename_map.items():
            old_path = os.path.join(folder_path, short_name)
            new_path = os.path.join(folder_path, long_name)
            if os.path.isfile(old_path):
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")