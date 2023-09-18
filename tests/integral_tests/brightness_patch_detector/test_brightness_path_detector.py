import cv2
import numpy as np
from brightness_patch_detector.brightness_path_detector import process_image

def test_integration():
    # 1. Load an image (Use a test image, preferably white 100x100 for simplicity)
    img_path = 'tests/test_images/test_image.jpg'
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 2. Process the image
    processed_path = 'tests/test_images/output_test_image.png'
    process_image(img_path, output_file_path=processed_path)
    processed_img = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)

    # 3. Verify the changes. We'll check if the processed image has some red pixels.
    # Remember, when reading with OpenCV in grayscale, red pixels might not necessarily have a value of 255 due to the color conversion.
    # For simplicity, if the average brightness of the processed image is greater than the original,
    # we can assume that a brighter quadrilateral was drawn on it.
    assert np.mean(processed_img) > np.mean(original_img), "The processed image does not seem to have the expected changes."