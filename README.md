# Tool Chest  

Simple repository containing code that solves basic problems conveniently without having to use the internet! 

# Face Centerer

A Python script that takes an image and creates a squared, pixelated retro-style image with the face centered. Supports PNG and JPG/JPEG image formats.

## Setup

1. Create a conda environment with the necessary dependencies:
   ```
   conda create -n img_processor python=3.9 -y
   conda activate img_processor
   conda install -y opencv numpy matplotlib pillow
   ```

## Usage

The script processes an image to create a pixelated retro-style image with the face centered:

```
python face_centerer.py path/to/your/image.jpg
```

### Supported Formats

The script supports common image formats:
- JPEG/JPG
- PNG

### Options

- `--output` or `-o`: Specify the output path for the processed image (default: input_filename_pixelated.jpg)
- `--size` or `-s`: Specify the size of the output square image in pixels (default: 256)
- `--pixel-size` or `-p`: Size of pixels (larger = more pixelated, default: 8)
- `--bits` or `-b`: Bits per color channel (1-5, lower = fewer colors, default: 3)
- `--pixel-method` or `-pm`: Pixelation method (`standard`, `adaptive`, or `area_based`, default: `adaptive`)
- `--color-method` or `-cm`: Color reduction method (`standard` or `kmeans`, default: `standard`)
- `--blend`: Blend factor with original image, 0.0-1.0 (default: 0.0)
- `--verbose` or `-v`: Enable verbose logging output (default: off)

### Examples

Process a JPEG image with default settings:
```
python face_centerer.py portrait.jpg
```

Create a highly pixelated 8-bit style image:
```
python face_centerer.py portrait.jpg --pixel-size 32 --bits 1 --output retro_portrait.jpg
```

Create a more recognizable pixelated image by using adaptive pixelation:
```
python face_centerer.py portrait.jpg --pixel-size 12 --pixel-method adaptive --bits 3
```

Create a nicely balanced retro effect with some original details preserved:
```
python face_centerer.py portrait.jpg --pixel-size 16 --bits 3 --blend 0.2
```

Enable verbose logging to see detailed processing information:
```
python face_centerer.py portrait.jpg --verbose
```

## Features

- Face detection using OpenCV's Haar Cascade Classifier
- Creates a square crop centered on the detected face
- Applies appropriate margins around the face
- Converts the image to a pixelated retro style using multiple algorithms:
  - Multiple pixelation methods to customize the retro effect
  - Adaptive pixelation that preserves facial features
  - Color quantization for retro-style color reduction
  - Optional blending with original for better recognition
- Customizable pixel size and color depth
- Falls back to image center if no face is detected
- Automatically locates or downloads the required face detection cascade file
- Robust logging system with configurable verbosity

## Pixelation Methods

- **Standard**: Traditional downsampling and upsampling for uniform pixelation
- **Adaptive**: Uses smaller pixels in the face region to preserve facial features
- **Area-based**: Calculates average color for fixed-size grid cells

## Color Reduction Methods

- **Standard**: Simple color quantization by reducing bits per channel
- **K-means**: Uses clustering to find optimal color palette

## Pixel Size and Color Depth

- **Pixel Size**: Controls the level of pixelation. Higher values create larger, more visible pixels.
  - A value of 4 creates subtle pixelation
  - A value of 8-16 creates a retro game-style pixelation
  - A value of 32 or higher creates extreme pixelation

- **Color Depth**: Controls the number of colors per channel.
  - 1 bit = 2 levels per channel (8 total colors) - extreme retro look
  - 2 bits = 4 levels per channel (64 total colors) - classic 8-bit style
  - 3 bits = 8 levels per channel (512 total colors) - 16-bit console style
  - 4-5 bits = more modern retro aesthetics

## Blend Factor

The blend factor allows you to mix the pixelated image with the original image:
- 0.0: Pure pixelated effect (default)
- 0.1-0.3: Subtle blend that improves recognition while maintaining pixelation
- 0.5: Equal mix of pixelated and original
- 0.7-0.9: Mostly original with subtle pixelation effect

## Logging Features

The script uses Python's built-in logging module for informative output:
- **Default**: Shows basic info, warnings, and errors
- **Verbose Mode**: Enable with `--verbose` to see additional debug information
- **Log Levels**:
  - INFO: General processing steps and results
  - WARNING: Non-critical issues that might affect output
  - ERROR: Problems that prevent successful processing
  - DEBUG: Detailed information (only in verbose mode)

## Troubleshooting

If the script can't locate the Haar cascade file, it will attempt to:
1. Search in common system locations
2. Download the file from the OpenCV GitHub repository

If your image isn't being read properly, ensure it's in either JPG/JPEG or PNG format.

## Tips for Best Results

1. For recognizable faces with retro aesthetics:
   - Use `--pixel-method adaptive` to preserve facial features
   - Use moderate pixel size (6-12)
   - Use higher color depth (`--bits 3` or `--bits 4`)
   - Add slight blending (`--blend 0.1` to `--blend 0.2`)

2. For classic 8-bit style:
   - Use `--pixel-method standard`
   - Use larger pixel size (12-24)
   - Use lower color depth (`--bits 2`)
   - Use no blending (`--blend 0.0`)

3. For troubleshooting:
   - Use `--verbose` to get more detailed information about processing
   - Check log messages for warnings or errors

## Requirements

- Python 3.9+
- OpenCV
- NumPy
- PIL/Pillow (as fallback for reading certain images)
