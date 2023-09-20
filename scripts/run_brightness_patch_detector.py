#!/usr/bin/python3
import argparse
import sys
from pathlib import Path

from brightness_patch_detector.brightness_path_detector import detect_and_draw_quadrilateral

if __name__ == "__main__":
    """
        Entry point for the brightness patch detector. 
        Processes the given image to find the quadrilateral formed by the centers of the four non-overlapping 
        5x5 patches with the highest average brightness. Then, draws this quadrilateral on the image.

        :param image_path: Path to the input image file.
        :param output_path: Path to the output image file.
        """
    if len(sys.argv) != 3:
        print("Usage: python brightness_patch_detector.py <path_to_image> <output_image_path>")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Command line arguments demo")

    # Define the obligatory argument
    parser.add_argument("image_path",
                        type=str,
                        help="Path to the input image file")

    # Define the optional argument with a default value
    parser.add_argument("output_path",
                        type=str,
                        default="output.png",
                        help="Path to the output image file")

    args = parser.parse_args()
    image_path = args.image_path
    output_path = args.output_path
    calculated_area = detect_and_draw_quadrilateral(image_path, output_path)
    print("Area of the quadrilateral:", calculated_area)
