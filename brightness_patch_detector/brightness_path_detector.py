from typing import List

import cv2
import numpy as np

from utils.decorators import error_handler

KERNEL_SIZE = 5
NUM_TOP_BRIGHT_PATCHES = 4
PATCH_CENTER_OFFSET = KERNEL_SIZE // 2
MIN_DISTANCE = 1


def get_average_brightness(image: np.ndarray, x: int, y: int) -> float:
    return float(np.mean(image[y:y + KERNEL_SIZE, x:x + KERNEL_SIZE]))


def get_patch_centers_and_brightness(image: np.ndarray, height: int, width: int) -> List:
    return [((x + PATCH_CENTER_OFFSET, y + PATCH_CENTER_OFFSET), get_average_brightness(image, x, y))
            for x in range(width - KERNEL_SIZE)
            for y in range(height - KERNEL_SIZE)]


def get_patches_sorted_by_brightness(image: np.ndarray, height: int, width: int) -> List:
    return sorted(get_patch_centers_and_brightness(image, height, width), key=lambda x: x[1], reverse=True)


def get_centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def get_points_sort_around_centroid(points: np.ndarray) -> np.ndarray:
    centroid = get_centroid(points)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[np.argsort(angles)]


def get_selected_patches(sorted_patches: List, grid: np.ndarray, height: int, width: int) -> np.ndarray:
    selected_coordinates = []

    for coordinates, _ in sorted_patches:
        patch_column = coordinates[0] // KERNEL_SIZE
        patch_row = coordinates[1] // KERNEL_SIZE

        y_start, y_end = max(patch_row - MIN_DISTANCE, 0), min(patch_row + MIN_DISTANCE + 1, height // KERNEL_SIZE)
        x_start, x_end = max(patch_column - MIN_DISTANCE, 0), min(patch_column + MIN_DISTANCE + 1, width // KERNEL_SIZE)

        if not np.any(grid[y_start:y_end, x_start:x_end]):
            selected_coordinates.append(coordinates)
            grid[y_start:y_end, x_start:x_end] = True

            if len(selected_coordinates) == NUM_TOP_BRIGHT_PATCHES:
                break

    return get_points_sort_around_centroid(np.array(selected_coordinates, dtype=int))


def get_grid(height: int, width: int) -> np.ndarray:
    return np.zeros((height // KERNEL_SIZE, width // KERNEL_SIZE), dtype=bool)


def get_top_patches(image: np.ndarray) -> np.ndarray:
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
    forward_diagonal_product = x_coordinates * np.roll(y_coordinates, -1)
    backward_diagonal_product = y_coordinates * np.roll(x_coordinates, -1)

    return 0.5 * abs(np.sum(forward_diagonal_product - backward_diagonal_product))


def get_area_of_quadrilateral(points: np.ndarray) -> float:
    height_coordinates = points[:, 0]
    width_coordinates = points[:, 1]

    return get_area_using_shoelace_formula(height_coordinates, width_coordinates)


def draw_and_save(image: np.ndarray, points: np.ndarray, output_path: str) -> None:
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
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        print(f"Error: Could not read image from path {image_path}. Please check the path and try again.")
        return

    top_patches = get_top_patches(gray_image)
    print("Area of the quadrilateral:", get_area_of_quadrilateral(top_patches))
    draw_and_save(gray_image, top_patches, output_path)
