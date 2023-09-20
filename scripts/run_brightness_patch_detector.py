#!/usr/bin/python3
import argparse
import sys
from pathlib import Path

from brightness_patch_detector.brightness_path_detector import draw_quadrilateral_and_calculate_area

if __name__ == "__main__":
    """
        Entry point for the brightness patch detector. 
        Processes the given image to find the quadrilateral formed by the centers of the four non-overlapping 
        5x5 patches with the highest average brightness. Then, draws this quadrilateral on the image.

        :param image_path: Path to the input image file.
        :param output_path: Path to the output image file.
        """
    parser = argparse.ArgumentParser(description="Brightness patch detector")

    parser.add_argument("image_path",
                        type=str,
                        help="Path to the input image file")

    parser.add_argument("--output",
                        dest="output_path",
                        type=str,
                        default="output.png",
                        nargs='?',
                        help="Path to the output image file (default: output.png)")

    args = parser.parse_args()

    if not args.image_path:
        print("Usage: python brightness_patch_detector.py <path_to_image> [--output <output_image_path>]")
        sys.exit(1)

    args = parser.parse_args()
    image_path = args.image_path
    output_path = args.output_path
    calculated_area = draw_quadrilateral_and_calculate_area(image_path, output_path)
    print("Area of the quadrilateral:", calculated_area)
