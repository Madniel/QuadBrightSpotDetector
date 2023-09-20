from typing import List

import cv2
import numpy as np

from utils.decorators import error_handler

KERNEL_SIZE = 5
NUM_TOP_BRIGHT_PATCHES = 4
PATCH_CENTER_OFFSET = KERNEL_SIZE // 2
MIN_DISTANCE_PREVENT_OVERLAP = KERNEL_SIZE - 1


def get_average_brightness(image: np.ndarray, x: int, y: int) -> float:
    """
    Calculate the average brightness of a patch.

    :param image: Input image.
    :param x: x-coordinate of the top-left corner of the patch.
    :param y: y-coordinate of the top-left corner of the patch.
    :return: Average brightness of the patch.
    """
    return float(np.mean(image[y:y + KERNEL_SIZE, x:x + KERNEL_SIZE]))


def get_patch_centers_and_brightness(image: np.ndarray, height: int, width: int) -> List:
    """
    Get center coordinates and average brightness for all patches.

    :param image: Input image.
    :param height: Image height.
    :param width: Image width.
    :return: List of center coordinates and corresponding brightness values.
    """
    return [((x + PATCH_CENTER_OFFSET, y + PATCH_CENTER_OFFSET), get_average_brightness(image, x, y))
            for x in range(width - KERNEL_SIZE)
            for y in range(height - KERNEL_SIZE)]


def get_patches_sorted_by_brightness(image: np.ndarray, height: int, width: int) -> List:
    """
    Get patches sorted by their brightness.

    :param image: Input image.
    :param height: Image height.
    :param width: Image width.
    :return: List of patches sorted by brightness.
    """
    return sorted(get_patch_centers_and_brightness(image, height, width), key=lambda x: x[1], reverse=True)


def get_centroid(points: np.ndarray) -> np.ndarray:
    """
    Calculate the centroid of a set of points.

    :param points: Set of points.
    :return: Centroid of the points.
    """
    return np.mean(points, axis=0)


def get_points_sort_around_centroid(points: np.ndarray) -> np.ndarray:
    """
    Sort points counter-clockwise around their centroid.

    :param points: Set of points.
    :return: Points sorted counter-clockwise.
    """
    centroid = get_centroid(points)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[np.argsort(angles)]


def get_selected_patches(sorted_patches: List, grid: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Select patches ensuring they don't overlap.

    :param sorted_patches: Patches sorted by brightness.
    :param grid: Grid to track selected patches.
    :param height: Image height.
    :param width: Image width.
    :return: Coordinates of selected patches.
    """
    selected_coordinates = np.empty((len(sorted_patches), 2), dtype=int)
    num_selected = 0

    height_factor = height // KERNEL_SIZE
    width_factor = width // KERNEL_SIZE

    for coordinates, _ in sorted_patches:
        patch_column = coordinates[0] // KERNEL_SIZE
        patch_row = coordinates[1] // KERNEL_SIZE

        x_start = max(patch_column - MIN_DISTANCE_PREVENT_OVERLAP, 0)
        x_end = min(patch_column + MIN_DISTANCE_PREVENT_OVERLAP + 1, width_factor)

        y_start = max(patch_row - MIN_DISTANCE_PREVENT_OVERLAP, 0)
        y_end = min(patch_row + MIN_DISTANCE_PREVENT_OVERLAP + 1, height_factor)

        if not np.any(grid[y_start:y_end, x_start:x_end]):
            selected_coordinates[num_selected] = coordinates
            num_selected += 1

            grid[y_start:y_end, x_start:x_end] = True

            if num_selected == NUM_TOP_BRIGHT_PATCHES:
                break

    return get_points_sort_around_centroid(selected_coordinates)


def get_grid(height: int, width: int) -> np.ndarray:
    """
    Generate a grid based on image dimensions and kernel size.

    :param height: Image height.
    :param width: Image width.
    :return: Grid for tracking selected patches.
    """
    return np.zeros((height // KERNEL_SIZE, width // KERNEL_SIZE), dtype=bool)


def get_top_patches(image: np.ndarray) -> np.ndarray:
    """
    Identify top bright patches in the image.

    :param image: Input image.
    :return: Coordinates of top patches.
    """
    height, width = image.shape
    patches_sorted_by_brightness = get_patches_sorted_by_brightness(image=image,
                                                                    height=height,
                                                                    width=width, )

    grid = get_grid(height=height, width=width)

    return get_selected_patches(sorted_patches=patches_sorted_by_brightness,
                                grid=grid,
                                height=height,
                                width=width, )


def get_area_using_shoelace_formula(x_coordinates: np.ndarray, y_coordinates: np.ndarray) -> float:
    """
    Calculate area using the shoelace formula.

    :param x_coordinates: x-coordinates of the polygon vertices.
    :param y_coordinates: y-coordinates of the polygon vertices.
    :return: Area of the polygon.
    """
    forward_diagonal_product = x_coordinates * np.roll(y_coordinates, -1)
    backward_diagonal_product = y_coordinates * np.roll(x_coordinates, -1)

    return 0.5 * abs(np.sum(forward_diagonal_product - backward_diagonal_product))


def get_area_of_quadrilateral(points: np.ndarray) -> float:
    """
    Compute the area of a quadrilateral defined by given points.

    :param points: Coordinates of the quadrilateral vertices.
    :return: Area of the quadrilateral.
    """
    height_coordinates = points[:, 0]
    width_coordinates = points[:, 1]

    return get_area_using_shoelace_formula(height_coordinates, width_coordinates)


def draw_and_save(image: np.ndarray, points: np.ndarray, output_path: str) -> None:
    """
    Draw a polygon on the image and save the result.

    :param image: Input image.
    :param points: Coordinates of the polygon vertices.
    :param output_path: Path to save the output image.
    """
    image_copy = image.copy()
    image_colored = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)

    points = points.reshape((1, -1, 1, 2))

    cv2.polylines(image_colored, points, isClosed=True, color=(0, 0, 255), thickness=2)
    success = cv2.imwrite(output_path, image_colored)

    if success:
        print(f"The image was saved successfully as {output_path}")
    else:
        print(f"Failed to save the image as {output_path}")


@error_handler
def process_image(image_path: str, output_path: str) -> None:
    """
    Process an image to detect, compute area, and visualize top bright patches.

    :param image_path: Path to the input image.
    :param output_path: Path to save the processed image.
    """
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        print(f"Error: Could not read image from path {image_path}. Please check the path and try again.")
        return

    top_patches = get_top_patches(gray_image)
    print("Area of the quadrilateral:", get_area_of_quadrilateral(top_patches))
    draw_and_save(gray_image, top_patches, output_path)
