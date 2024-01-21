#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset/nerf_synthetic"

# Iterate over each directory in the specified path
for dir in "$path_to_directories"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        python convert.py --source_path $dir --colmap_executable /usr/bin/colmap
    fi
done
