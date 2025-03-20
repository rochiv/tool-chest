#!/usr/bin/env python
import cv2
import numpy as np
import argparse
import os
import sys
import logging
import urllib.request
from pathlib import Path

# Set up logging
logger = logging.getLogger("face_centerer")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Try to import PIL for alternative image reading
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logger.debug("PIL not available for fallback image reading")

def find_haarcascade_file():
    """Find the haarcascade file location"""
    cascade_filename = 'haarcascade_frontalface_default.xml'
    
    # Check in common locations
    common_locations = [
        # Current directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), cascade_filename),
        # Common system paths
        os.path.join(sys.prefix, 'share', 'opencv4', 'haarcascades', cascade_filename),
        os.path.join(sys.prefix, 'share', 'opencv', 'haarcascades', cascade_filename),
        # Anaconda/conda paths
        os.path.join(sys.prefix, 'lib', 'python{}.{}'.format(sys.version_info[0], sys.version_info[1]), 
                    'site-packages', 'cv2', 'data', 'haarcascades', cascade_filename),
        # Additional common paths on various systems
        '/usr/share/opencv/haarcascades/' + cascade_filename,
        '/usr/local/share/opencv/haarcascades/' + cascade_filename,
        '/opt/local/share/opencv/haarcascades/' + cascade_filename,
    ]
    
    # Try each location
    for path in common_locations:
        if os.path.exists(path):
            logger.info(f"Found cascade file at: {path}")
            return path
            
    # If not found, download it
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cascade_filename)
    logger.warning(f"Cascade file not found in common locations. Downloading to {local_path}")
    
    try:
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, local_path)
        
        if os.path.exists(local_path):
            logger.info("Download successful!")
            return local_path
    except Exception as e:
        logger.error(f"Error downloading cascade file: {e}")
    
    raise FileNotFoundError("Could not find or download haarcascade_frontalface_default.xml")

def is_supported_image(file_path):
    """Check if the file is a supported image type (PNG or JPG/JPEG)"""
    file_path = Path(file_path)
    supported_extensions = ['.jpg', '.jpeg', '.png']
    return file_path.suffix.lower() in supported_extensions

def pixelate_image(image, pixel_size, method='adaptive'):
    """
    Pixelate an image by downsampling and then upsampling without interpolation.
    
    Args:
        image: Input image
        pixel_size: Size of pixels (larger = more pixelated)
        method: Pixelation method ('standard', 'adaptive', or 'area_based')
    
    Returns:
        Pixelated image
    """
    height, width = image.shape[:2]
    
    if method == 'standard':
        # Standard pixelation (uniform across the image)
        small_width = max(1, width // pixel_size)
        small_height = max(1, height // pixel_size)
        temp = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
        
    elif method == 'adaptive':
        # Adaptive pixelation (preserves face details)
        face_region_y = height // 3
        face_region_height = height // 3
        
        # Process different regions with different levels of pixelation
        pixelated = np.copy(image)
        
        # Top portion - more pixelated
        top_portion = image[:face_region_y, :]
        if top_portion.size > 0:
            top_small = cv2.resize(top_portion, 
                                (max(1, top_portion.shape[1] // pixel_size), 
                                 max(1, top_portion.shape[0] // pixel_size)), 
                                interpolation=cv2.INTER_LINEAR)
            top_pixelated = cv2.resize(top_small, (top_portion.shape[1], top_portion.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            pixelated[:face_region_y, :] = top_pixelated
        
        # Face region - less pixelated
        face_portion = image[face_region_y:face_region_y+face_region_height, :]
        if face_portion.size > 0:
            face_small = cv2.resize(face_portion, 
                                  (max(1, face_portion.shape[1] // max(1, pixel_size//2)), 
                                   max(1, face_portion.shape[0] // max(1, pixel_size//2))), 
                                  interpolation=cv2.INTER_LINEAR)
            face_pixelated = cv2.resize(face_small, (face_portion.shape[1], face_portion.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            pixelated[face_region_y:face_region_y+face_region_height, :] = face_pixelated
        
        # Bottom portion - more pixelated
        bottom_portion = image[face_region_y+face_region_height:, :]
        if bottom_portion.size > 0:
            bottom_small = cv2.resize(bottom_portion, 
                                    (max(1, bottom_portion.shape[1] // pixel_size), 
                                     max(1, bottom_portion.shape[0] // pixel_size)), 
                                    interpolation=cv2.INTER_LINEAR)
            bottom_pixelated = cv2.resize(bottom_small, (bottom_portion.shape[1], bottom_portion.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            pixelated[face_region_y+face_region_height:, :] = bottom_pixelated
        
    elif method == 'area_based':
        # Area-based pixelation
        pixelated = np.copy(image)
        for y in range(0, height, pixel_size):
            for x in range(0, width, pixel_size):
                h = min(pixel_size, height - y)
                w = min(pixel_size, width - x)
                if h > 0 and w > 0:  # Ensure we have valid dimensions
                    region = image[y:y+h, x:x+w]
                    color = np.mean(region, axis=(0, 1)).astype(np.uint8)
                    pixelated[y:y+h, x:x+w] = color
    
    else:
        # Fallback to standard method
        logger.warning(f"Unknown pixelation method '{method}', falling back to standard")
        small_width = max(1, width // pixel_size)
        small_height = max(1, height // pixel_size)
        temp = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    
    return pixelated

def reduce_colors(image, bits_per_channel=3, method='standard'):
    """
    Reduce the color depth of an image.
    
    Args:
        image: Input image
        bits_per_channel: Number of bits to keep per color channel
        method: Color reduction method ('standard' or 'kmeans')
    
    Returns:
        Image with reduced color depth
    """
    if method == 'standard':
        # Standard color quantization
        levels = 2 ** bits_per_channel
        factor = 256 // levels
        quantized = (image // factor) * factor
        
    elif method == 'kmeans':
        # K-means color clustering
        try:
            # Reshape the image to a 2D array of pixels
            h, w, c = image.shape
            pixels = image.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            # Define criteria and apply kmeans
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            k = min(2 ** (bits_per_channel * 3), pixels.shape[0])  # Cap k at the number of pixels
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to 8-bit values
            centers = np.uint8(centers)
            
            # Map labels to colors
            quantized = centers[labels.flatten()]
            
            # Reshape back to original image format
            quantized = quantized.reshape(image.shape)
        except Exception as e:
            logger.error(f"K-means color reduction failed: {e}")
            logger.info("Falling back to standard color reduction")
            # Fall back to standard method
            levels = 2 ** bits_per_channel
            factor = 256 // levels
            quantized = (image // factor) * factor
    else:
        # Fallback to standard method
        logger.warning(f"Unknown color reduction method '{method}', falling back to standard")
        levels = 2 ** bits_per_channel
        factor = 256 // levels
        quantized = (image // factor) * factor
    
    return quantized

def read_image_with_pil(image_path):
    """Read an image using PIL and convert to OpenCV format"""
    if not PILLOW_AVAILABLE:
        logger.debug("PIL not available for image reading")
        return None
    
    try:
        # Open image with PIL
        pil_img = Image.open(image_path)
        # Convert to RGB if it's not already
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        # Convert to numpy array
        img_array = np.array(pil_img)
        # Convert RGB to BGR (OpenCV format)
        opencv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return opencv_img
    except Exception as e:
        logger.error(f"PIL image reading failed: {e}")
        return None

def process_image(image_path, output_path=None, size=256, pixel_size=8, bits_per_channel=3, 
                 pixel_method='adaptive', color_method='standard', blend_factor=0.0):
    """
    Process an image to create a squared pixelated image with face centered.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Path to save the output image
        size (int): Size of the output square image
        pixel_size (int): Size of pixels (larger = more pixelated)
        bits_per_channel (int): Bits per color channel for color quantization
        pixel_method (str): Pixelation method ('standard', 'adaptive', or 'area_based')
        color_method (str): Color reduction method ('standard' or 'kmeans')
        blend_factor (float): How much of the original image to blend (0.0-1.0)
    
    Returns:
        np.ndarray: Processed image
    """
    # Check if the image format is supported
    if not is_supported_image(image_path):
        raise ValueError(f"Unsupported image format. Only PNG and JPG/JPEG are supported.")
    
    # Try reading the image with OpenCV
    img = cv2.imread(image_path)
    
    # If OpenCV fails, try with PIL
    if img is None:
        logger.info(f"OpenCV couldn't read the image. Trying with PIL...")
        img = read_image_with_pil(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Load the pre-trained face detector
    cascade_file = find_haarcascade_file()
    face_cascade = cv2.CascadeClassifier(cascade_file)
    
    if face_cascade.empty():
        raise ValueError(f"Failed to load face cascade classifier from {cascade_file}")
    
    # Detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        logger.info("No faces detected. Using the center of the image.")
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        face_w, face_h = w // 4, h // 4  # Default face size if none detected
    else:
        # Use the first face detected
        x, y, w, h = faces[0]
        center_x, center_y = x + w // 2, y + h // 2
        face_w, face_h = w, h
        logger.info(f"Face detected at position ({x}, {y}) with size {w}x{h}")
    
    # Determine the square crop size (larger of face width or height, with a margin)
    crop_size = int(max(face_w, face_h) * 2.5)  # Increased margin for better framing
    
    # Calculate crop boundaries
    left = max(0, int(center_x - crop_size // 2))
    top = max(0, int(center_y - crop_size // 2))
    right = min(img.shape[1], int(center_x + crop_size // 2))
    bottom = min(img.shape[0], int(center_y + crop_size // 2))
    
    # Adjust crop to ensure it's square
    crop_width = right - left
    crop_height = bottom - top
    
    if crop_width > crop_height:
        # Increase height
        diff = crop_width - crop_height
        top = max(0, int(top - diff // 2))
        bottom = min(img.shape[0], int(bottom + diff // 2))
    elif crop_height > crop_width:
        # Increase width
        diff = crop_height - crop_width
        left = max(0, int(left - diff // 2))
        right = min(img.shape[1], int(right + diff // 2))
    
    # Ensure all values are integers
    top, bottom, left, right = int(top), int(bottom), int(left), int(right)
    
    # Crop the image
    cropped_img = img[top:bottom, left:right]
    
    # Resize to the specified size
    resized_img = cv2.resize(cropped_img, (size, size))
    
    # Save a copy of the original (resized) image for potential blending
    original_resized = resized_img.copy()
    
    # Apply pixelation effect
    pixelated_img = pixelate_image(resized_img, pixel_size, pixel_method)
    
    # Reduce colors for retro look
    retro_img = reduce_colors(pixelated_img, bits_per_channel, color_method)
    
    # Apply blending if requested
    if blend_factor > 0:
        blended_img = cv2.addWeighted(retro_img, 1 - blend_factor, original_resized, blend_factor, 0)
        retro_img = blended_img
    
    # Convert to 32-bit floating point format
    img_32bit = retro_img.astype(np.float32)
    
    # Save the result if output path is provided
    if output_path:
        cv2.imwrite(output_path, retro_img)
        logger.info(f"Processed image saved to {output_path}")
    
    return img_32bit

def main():
    parser = argparse.ArgumentParser(description='Process an image to create a pixelated retro image with face centered.')
    parser.add_argument('input_image', help='Path to the input image (PNG or JPG/JPEG only)')
    parser.add_argument('--output', '-o', help='Path to save the output image')
    parser.add_argument('--size', '-s', type=int, default=256, help='Size of the output square image (default: 256)')
    parser.add_argument('--pixel-size', '-p', type=int, default=8, help='Size of pixels (larger = more pixelated, default: 8)')
    parser.add_argument('--bits', '-b', type=int, default=3, help='Bits per color channel (1-5, lower = fewer colors, default: 3)')
    parser.add_argument('--pixel-method', '-pm', choices=['standard', 'adaptive', 'area_based'], default='adaptive', 
                        help='Pixelation method (default: adaptive)')
    parser.add_argument('--color-method', '-cm', choices=['standard', 'kmeans'], default='standard', 
                        help='Color reduction method (default: standard)')
    parser.add_argument('--blend', type=float, default=0.0, 
                        help='Blend factor with original image, 0.0-1.0 (default: 0.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Validate bits per channel
    if args.bits < 1 or args.bits > 5:
        logger.warning(f"Bits per channel must be between 1 and 5. Setting to default (3).")
        args.bits = 3
    
    # Validate blend factor
    if args.blend < 0.0 or args.blend > 1.0:
        logger.warning(f"Blend factor must be between 0.0 and 1.0. Setting to default (0.0).")
        args.blend = 0.0
    
    # Generate default output path if not provided
    if args.output is None:
        base_name = os.path.basename(args.input_image)
        name, ext = os.path.splitext(base_name)
        args.output = f"{name}_pixelated.jpg"  # Always use jpg for output
    
    try:
        # Process the image
        processed_img = process_image(
            args.input_image, 
            args.output, 
            args.size, 
            args.pixel_size, 
            args.bits, 
            args.pixel_method, 
            args.color_method,
            args.blend
        )
        logger.info(f"Successfully processed image to pixelated {2**args.bits}x{2**args.bits}x{2**args.bits}-color image")
        logger.debug(f"Image dimensions: {processed_img.shape}")
        logger.debug(f"Image data type: {processed_img.dtype}")
        logger.info(f"Pixel size: {args.pixel_size}px")
        logger.info(f"Pixelation method: {args.pixel_method}")
        logger.info(f"Color method: {args.color_method}")
        logger.info(f"Color depth: {args.bits} bits per channel ({2**args.bits} levels per channel)")
        if args.blend > 0:
            logger.info(f"Blend factor: {args.blend}")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
