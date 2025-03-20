#!/bin/bash

# Set the folder containing the zip files
folder_path="/home/rohit/Code/AutoPET/FDG-PET-CT-Lesions"

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
  echo "Error: Folder not found!"
  exit 1
fi

# Navigate to the folder
cd "$folder_path"

# Unzip all the files in the folder
unzip '*.zip'

# Remove the zip files
rm -f *.zip

echo "Unzipping and file deletion complete!"
