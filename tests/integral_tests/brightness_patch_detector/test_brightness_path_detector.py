import tempfile
from pathlib import Path

import cv2
import numpy as np

from brightness_patch_detector.brightness_path_detector import process_image


def test_process_image():
    resources_path = Path(__file__).parent.parent.parent / "resources" / "brightness_patch_detector"
    test_image_path = resources_path / "image.jpg"
    ground_truth_path = resources_path / "output_ground_truth.png"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_directory = Path(temp_dir)
        test_output_path = temp_directory / "output.png"

        process_image(str(test_image_path), str(test_output_path))

        assert test_output_path.exists(), f"Output image not saved at {test_output_path}"

        processed_img = cv2.imread(str(test_output_path))
        ground_truth_img = cv2.imread(str(ground_truth_path))

        assert processed_img is not None, "Unable to load the processed image."
        assert ground_truth_img is not None, "Unable to load the ground truth image."

        difference = np.abs(processed_img.astype(int) - ground_truth_img.astype(int))
        assert np.sum(difference) == 0, "Processed image differs from the ground truth."
