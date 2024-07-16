#!/bin/bash

python convert.py --source_path datasets/cube --colmap_executable /opt/homebrew/bin/colmap --camera OPENCV_FISHEYE
cd datasets/cube
mkdir fish
cp -r input fish
cp -r distorted/sparse fish
mv fish/input fish/images
cd ../..
