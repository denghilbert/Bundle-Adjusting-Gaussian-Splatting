#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset/FisheyeNeRF"

# Iterate over each directory in the specified path
for dir in "$path_to_directories"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        #echo python render.py -s $dir -m "baseline/vanillar_$name" --iteration 30000
        #python render.py -s $dir -m "baseline/vanilla_$name" --iteration 30000
        echo python render.py -s $dir -m "baseline/grid_$name" --iteration 30000
        python render.py -s $dir -m "baseline/grid_$name" --iteration 30000
        #echo python render.py -s $dir -m "baseline/ref_$name" --iteration 30000
        #python render.py -s $dir -m "baseline/ref_$name" --iteration 30000
    fi
done
