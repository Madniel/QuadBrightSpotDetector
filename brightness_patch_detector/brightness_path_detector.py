from typing import List, Tuple

import cv2
import numpy as np

from utils.decorators import error_handler
from utils.exceptions import ImageProcessingError
from utils.patch import Patch

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


def get_patch_centers_and_brightness(image: np.ndarray) -> List[Patch]:
    """
    Get center coordinates and average brightness for all patches.

    :param image: Input image.
    :return: List of center coordinates and corresponding brightness values.
    """
    height, width = image.shape
    patch_center_offset = KERNEL_SIZE // 2
    patches = []

    for y in range(height - KERNEL_SIZE):
        for x in range(width - KERNEL_SIZE):
            center_x = x + patch_center_offset
            center_y = y + patch_center_offset
            brightness = get_average_brightness(image, x, y)
            patches.append(Patch(center=(center_x, center_y), brightness=brightness))

    return patches


def get_patches_sorted_by_brightness(image: np.ndarray) -> List:
    """
    Get patches sorted by their brightness.

    :param image: Input image.
    :return: List of patches sorted by brightness.
    """
    patches = get_patch_centers_and_brightness(image)
    sorted_patches = sorted(patches, key=lambda patch: patch.brightness, reverse=True)
    return [patch.center for patch in sorted_patches]


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


def patch_coordinates_within_bounds(coordinates: Tuple[int, int],
                                    width_factor: int,
                                    height_factor: int) -> Tuple[int, int, int, int]:
    """
    Calculate the bounds for checking overlap.

    :param coordinates: Coordinates of the point
    :param width_factor: Factor of image width by kernel size.
    :param height_factor: Factor of image height by kernel size.
    :return: Tuple of start and end coordinates for both x and y directions.
    """
    patch_col = coordinates[0] // KERNEL_SIZE
    patch_row = coordinates[1] // KERNEL_SIZE
    x_start = max(patch_col - MIN_DISTANCE_PREVENT_OVERLAP, 0)
    x_end = min(patch_col + MIN_DISTANCE_PREVENT_OVERLAP + 1, width_factor)
    y_start = max(patch_row - MIN_DISTANCE_PREVENT_OVERLAP, 0)
    y_end = min(patch_row + MIN_DISTANCE_PREVENT_OVERLAP + 1, height_factor)
    return x_start, x_end, y_start, y_end


def is_place_patch_possible(grid: np.ndarray, x_start: int, x_end: int, y_start: int, y_end: int) -> bool:
    """
    Check if a patch can be placed at the specified location.

    :param grid: Grid tracking selected patches.
    :param x_start: Start x-coordinate.
    :param x_end: End x-coordinate.
    :param y_start: Start y-coordinate.
    :param y_end: End y-coordinate.
    :return: Boolean indicating if patch can be placed.
    """
    return not np.any(grid[y_start:y_end, x_start:x_end])


def get_selected_patches(image: np.ndarray, sorted_patches: List[Tuple[int, int]]) -> np.ndarray:
    """
    Select patches ensuring they don't overlap.

    :param image: Input image.
    :param sorted_patches: Patches sorted by brightness.
    :return: Coordinates of selected patches.
    """
    grid = get_grid(image)
    selected_coordinates = []
    height_factor = image.shape[0] // KERNEL_SIZE
    width_factor = image.shape[1] // KERNEL_SIZE

    for coordinates in sorted_patches:
        x_start, x_end, y_start, y_end = patch_coordinates_within_bounds(coordinates,
                                                                         width_factor,
                                                                         height_factor)

        if is_place_patch_possible(grid, x_start, x_end, y_start, y_end):
            selected_coordinates.append(coordinates)
            grid[y_start:y_end, x_start:x_end] = True

            if len(selected_coordinates) == NUM_TOP_BRIGHT_PATCHES:
                break

    selected_coordinates_array = np.array(selected_coordinates, dtype=int)
    return get_points_sort_around_centroid(selected_coordinates_array)


def get_grid(image: np.ndarray) -> np.ndarray:
    """
    Generate a grid based on image dimensions and kernel size.

    :param image: Input image.
    :return: Grid for tracking selected patches.
    """

    height, width = image.shape
    return np.zeros((height // KERNEL_SIZE, width // KERNEL_SIZE), dtype=bool)


def get_top_patches(image: np.ndarray) -> np.ndarray:
    """
    Identify top bright patches in the image.

    :param image: Input image.
    :return: Coordinates of top patches.
    """

    patches_sorted_by_brightness = get_patches_sorted_by_brightness(image=image)
    return get_selected_patches(image=image, sorted_patches=patches_sorted_by_brightness)


def get_area_of_quadrilateral(x_coordinates: np.ndarray, y_coordinates: np.ndarray) -> float:
    """
    Calculate area  of a quadrilateral using the shoelace formula.

    :param x_coordinates: x-coordinates of the polygon vertices.
    :param y_coordinates: y-coordinates of the polygon vertices.
    :return: Area of the polygon.
    """
    forward_diagonal_product = x_coordinates * np.roll(y_coordinates, -1)
    backward_diagonal_product = y_coordinates * np.roll(x_coordinates, -1)

    return 0.5 * abs(np.sum(forward_diagonal_product - backward_diagonal_product))


def get_image_with_polygon(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Draw a polygon on the image.

    :param image: Input image.
    :param points: Coordinates of the polygon vertices.
    """
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    points = points.reshape((1, -1, 1, 2))
    cv2.polylines(image_colored, points, isClosed=True, color=(0, 0, 255), thickness=2)
    return image_colored


def get_image_from_file(image_path: str):
    """
    Read an image from a given path in grayscale.

    :param image_path: Path to the input image.
    :return: Grayscale image.
    :raises ImageProcessingError: If the image cannot be read.
    """
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise ImageProcessingError(f"Could not read image from path {image_path}. Please check the path and try again.")
    return gray_image


def save_image_to_file(output_path: str, image) -> None:
    """
    Save an image to a given path.

    :param output_path: Path to save the processed image.
    :param image: Image to save.
    :raises ImageProcessingError: If the image cannot be saved.
    """
    is_image_saved = cv2.imwrite(output_path, image)
    if not is_image_saved:
        raise ImageProcessingError(f"Failed to save the image as {output_path}")


@error_handler
def draw_quadrilateral_and_calculate_area(image_path: str, output_path: str) -> float:
    """
    Process an image to detect, compute area, and visualize top bright patches.

    :param image_path: Path to the input image.
    :param output_path: Path to save the processed image.
    :return: Area of the quadrilateral.
    """
    gray_image = get_image_from_file(image_path)
    top_patches = get_top_patches(gray_image)
    image_colored = get_image_with_polygon(gray_image, top_patches)
    save_image_to_file(output_path, image_colored)
    height_coordinates = top_patches[:, 0]
    width_coordinates = top_patches[:, 1]
    return get_area_of_quadrilateral(height_coordinates, width_coordinates)
