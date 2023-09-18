import os
import tempfile

import numpy as np
import cv2
import pytest
from brightness_patch_detector.brightness_path_detector import get_top_patches, get_area_of_quadrilateral, process_image, \
    draw_and_save


@pytest.fixture
def sample_image():
    # This fixture creates a 100x100 grayscale image with a 10x10 bright spot in the center.
    image = np.zeros((100, 100), dtype=np.uint8)
    image[45:55, 45:55] = 255
    yield image


def test_find_top_patches(sample_image):
    patches = get_top_patches(sample_image)

    # Check that we found the patches
    assert len(patches) == 4

    # Since the bright spot is at the center, at least one of the patches should be near that point.
    assert (55, 55) in patches


def test_area_of_quadrilateral():
    # For a 2x2 square, the area should be 4
    points = [(0, 0), (0, 2), (2, 2), (2, 0)]
    assert get_area_of_quadrilateral(points) == 4.0

    # For a 1x2 rectangle, the area should be 2
    points = [(0, 0), (0, 2), (1, 2), (1, 0)]
    assert get_area_of_quadrilateral(points) == 2.0


def test_draw_and_save(sample_image):
    # Define mock points for the quadrilateral
    mock_points = [(10, 10), (90, 10), (90, 90), (10, 90)]

    # Use a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create path for the output image within the temporary directory
        output_file_path = os.path.join(temp_dir, "output.png")

        # Use draw_and_save to draw the quadrilateral and save it
        draw_and_save(sample_image, mock_points, output_file_path)

        # Read the saved image and validate
        saved_img = cv2.imread(output_file_path)

        # Check if there are red pixels
        red_pixels = (saved_img[:, :, 2] == 255) & (saved_img[:, :, 1] == 0) & (saved_img[:, :, 0] == 0)

        assert np.any(red_pixels), "The quadrilateral does not seem to be drawn on the image."


def test_process_image(sample_image):
    # Use a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create paths for the input and output images within the temporary directory
        input_file_path = os.path.join(temp_dir, "input.png")
        output_file_path = os.path.join(temp_dir, "output.png")

        # Save the sample_image to the input file path
        cv2.imwrite(input_file_path, sample_image)

        # Call the modified process_image function
        process_image(input_file_path, output_file_path)

        # Read the saved image from the output file path
        saved_img = cv2.imread(output_file_path)

        # Check if there are red pixels indicating the drawn quadrilateral
        red_pixels = (saved_img[:, :, 2] == 255) & (saved_img[:, :, 1] == 0) & (saved_img[:, :, 0] == 0)
        assert np.any(red_pixels), "The quadrilateral does not seem to be drawn on the image."