import os
import tempfile

import cv2
import numpy as np
import pytest

from brightness_patch_detector.brightness_path_detector import get_top_patches, get_area_of_quadrilateral, \
    draw_and_save, get_average_brightness, get_patch_centers_and_brightness, get_area_using_shoelace, get_patches_sorted_by_brightness, \
    get_centroid, get_points_sort_around_centroid, get_grid, get_selected_patches

NUMBER_PATCHES = 4
KERNEL_SIZE = 5


@pytest.fixture
def exemplary_image():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[45:55, 45:55] = 255
    yield image


def test_get_average_brightness(exemplary_image):
    brightness = get_average_brightness(exemplary_image, 40, 40)
    expected_value_black_patch = 0.0
    assert brightness == expected_value_black_patch, f"Expected {expected_value_black_patch}, but got {brightness}"

    brightness = get_average_brightness(exemplary_image, 45, 45)
    expected_value_white_patch = 255.0
    assert brightness == expected_value_white_patch, f"Expected {expected_value_white_patch}, but got {brightness}"

    brightness = get_average_brightness(exemplary_image, 44, 41)
    number_white_pixels = 4
    area = KERNEL_SIZE * KERNEL_SIZE
    expected_value_mixed_patch = 255 * number_white_pixels / area  # Only 4 pixels out of the 25 are white.
    assert brightness == expected_value_mixed_patch, f"Expected {expected_value_mixed_patch}, but got {brightness}"


def test_get_patches(exemplary_image):
    height, width = exemplary_image.shape
    patches = get_patch_centers_and_brightness(exemplary_image, height, width)
    assert len(patches) == (height - KERNEL_SIZE) * (width - KERNEL_SIZE)


def test_sorted_patches(exemplary_image):
    sorted_patches = get_patches_sorted_by_brightness(exemplary_image, 100, 100)

    brightness_values = [patch[1] for patch in sorted_patches]
    assert all(brightness_values[i] >= brightness_values[i + 1] for i in range(len(brightness_values) - 1))


def test_compute_centroid():
    square = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])

    centroid = get_centroid(square)
    assert np.allclose(centroid, [1, 1]), f"Expected [1, 1], but got {centroid}"

    triangle = np.array([[0, 0], [0, 2], [2, 0]])

    centroid = get_centroid(triangle)
    expected_centroid = [2 / 3, 2 / 3]
    assert np.allclose(centroid, expected_centroid), f"Expected {expected_centroid}, but got {centroid}"

    random_points = np.random.rand(10, 2)
    centroid = get_centroid(random_points)

    assert centroid.shape == (2,), "Centroid should have 2 coordinates"
    assert 0 <= centroid[0] <= 1, "Centroid x-coordinate out of bounds"
    assert 0 <= centroid[1] <= 1, "Centroid y-coordinate out of bounds"


def test_order_points():
    square = np.array([[2, 2], [0, 0], [2, 0], [0, 2]])

    ordered_square = get_points_sort_around_centroid(square)
    expected_order = np.array([[0, 0],
                               [2, 0],
                               [2, 2],
                               [0, 2]])
    assert np.array_equal(ordered_square, expected_order), f"Expected {expected_order}, but got {ordered_square}"

    triangle = np.array([[0, 2], [2, 0], [0, 0]])

    ordered_triangle = get_points_sort_around_centroid(triangle)
    expected_order = np.array([[0, 0],
                               [2, 0],
                               [0, 2]])
    assert np.array_equal(ordered_triangle, expected_order), f"Expected {expected_order}, but got {ordered_triangle}"

    points = np.random.rand(5, 2)

    ordered_points = get_points_sort_around_centroid(points)
    centroid = get_centroid(points)
    angles = np.arctan2(ordered_points[:, 1] - centroid[1], ordered_points[:, 0] - centroid[0])

    assert all(angles[i] <= angles[i + 1] for i in range(len(angles) - 1)), "Points are not ordered counter-clockwise"


def test_get_selected_patches(exemplary_image):
    sorted_patches = get_patches_sorted_by_brightness(image=exemplary_image,
                                                      height=exemplary_image.shape[0],
                                                      width=exemplary_image.shape[1])

    grid = get_grid(height=exemplary_image.shape[0], width=exemplary_image.shape[1])
    selected_patches = get_selected_patches(sorted_patches=sorted_patches,
                                            grid=grid,
                                            height=exemplary_image.shape[0],
                                            width=exemplary_image.shape[1])

    assert [47, 47] in selected_patches.tolist(), "Expected center patch to be selected"

    exemplary_image[10:15, 10:15] = 255
    exemplary_image[10:15, 85:90] = 255
    exemplary_image[85:90, 10:15] = 255
    exemplary_image[85:90, 85:90] = 255

    sorted_patches = get_patches_sorted_by_brightness(image=exemplary_image,
                                                      height=exemplary_image.shape[0],
                                                      width=exemplary_image.shape[1])

    grid = get_grid(height=exemplary_image.shape[0], width=exemplary_image.shape[1])
    selected_patches = get_selected_patches(sorted_patches=sorted_patches,
                                            grid=grid,
                                            height=exemplary_image.shape[0],
                                            width=exemplary_image.shape[1])

    expected_patches = [[12, 12],
                        [12, 87],
                        [87, 12],]
    assert all([patch in selected_patches.tolist() for patch in
                expected_patches]), f"Expected patches not found in selected patches"


def test_get_grid(exemplary_image):
    grid = get_grid(height=exemplary_image.shape[0], width=exemplary_image.shape[1])

    expected_shape = (exemplary_image.shape[0] // KERNEL_SIZE, exemplary_image.shape[1] // KERNEL_SIZE)
    assert grid.shape == expected_shape, f"Expected shape: {expected_shape}, but got: {grid.shape}"

    grid = get_grid(height=exemplary_image.shape[0], width=exemplary_image.shape[1])
    assert not np.any(grid), "Grid should consist of all False values"


def test_get_top_patches(exemplary_image):
    top_patches = get_top_patches(exemplary_image)
    exemplary_patches = (47, 47)

    assert len(top_patches) == NUMBER_PATCHES
    assert exemplary_patches in top_patches


def test_get_shoelace_formula_result():
    square_area = 4.0
    x_coordinates = np.array([1, 1, 3, 3])
    y_coordinates = np.array([1, 3, 3, 1])

    area = get_area_using_shoelace(x_coordinates, y_coordinates)
    assert area == square_area, f"Expected 4, but got {square_area}"

    rectangle_area = 12.0
    x_coordinates = np.array([2, 2, 6, 6])
    y_coordinates = np.array([1, 4, 4, 1])

    area = get_area_using_shoelace(x_coordinates, y_coordinates)
    assert area == rectangle_area, f"Expected 12, but got {rectangle_area}"

    triangle_area = 6.0
    x_coordinates = np.array([1, 1, 5])
    y_coordinates = np.array([1, 4, 1])

    area = get_area_using_shoelace(x_coordinates, y_coordinates)
    assert area == triangle_area, f"Expected 6, but got {triangle_area}"


def test_area_of_quadrilateral():
    square_corners = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
    assert get_area_of_quadrilateral(square_corners) == 4.0

    rectangle_corners = np.array([[0, 0], [0, 2], [1, 2], [1, 0]])
    assert get_area_of_quadrilateral(rectangle_corners) == 2.0


def test_draw_and_save(exemplary_image):
    quadrilateral_corners = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file_path = os.path.join(temp_dir, "output.png")
        draw_and_save(exemplary_image, quadrilateral_corners, output_file_path)
        saved_image = cv2.imread(output_file_path)
        red_pixels = (saved_image[:, :, 2] == 255) & (saved_image[:, :, 1] == 0) & (saved_image[:, :, 0] == 0)

        assert np.any(red_pixels), "The quadrilateral does not seem to be drawn on the image."



