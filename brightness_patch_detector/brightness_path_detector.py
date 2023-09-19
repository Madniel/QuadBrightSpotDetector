from typing import List

import cv2
import numpy as np

from utils.decorators import error_handler

KERNEL_SIZE = 5
NUMBER_PATCHES = 4
PATCH_CENTER_OFFSET = KERNEL_SIZE // 2
MIN_DISTANCE = 1


def get_average_brightness(image: np.ndarray, x: int, y: int) -> float:
    return np.mean(image[y:y + KERNEL_SIZE, x:x + KERNEL_SIZE]).astype(np.float32)


def get_patches(image: np.ndarray, height: int, width: int) -> List:
    return [((x + PATCH_CENTER_OFFSET, y + PATCH_CENTER_OFFSET), get_average_brightness(image, x, y))
            for x in range(width - KERNEL_SIZE)
            for y in range(height - KERNEL_SIZE)]


def get_sorted_patches(image: np.ndarray, height: int, width: int) -> List:
    return sorted(get_patches(image, height, width), key=lambda x: x[1], reverse=True)


def compute_centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def order_points(points: np.ndarray) -> np.ndarray:
    centroid = compute_centroid(points)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[np.argsort(angles)]


def get_selected_patches(sorted_patches: List, grid: np.ndarray, height: int, width: int) -> np.ndarray:
    selected_coordinates = []

    for coordinates, _ in sorted_patches:
        grid_x = coordinates[0] // KERNEL_SIZE
        grid_y = coordinates[1] // KERNEL_SIZE

        y_start, y_end = max(grid_y - MIN_DISTANCE, 0), min(grid_y + MIN_DISTANCE + 1, height // KERNEL_SIZE)
        x_start, x_end = max(grid_x - MIN_DISTANCE, 0), min(grid_x + MIN_DISTANCE + 1, width // KERNEL_SIZE)

        if not np.any(grid[y_start:y_end, x_start:x_end]):
            selected_coordinates.append(coordinates)
            grid[y_start:y_end, x_start:x_end] = True

            if len(selected_coordinates) == NUMBER_PATCHES:
                break

    return order_points(np.array(selected_coordinates, dtype=int))


def get_grid(height: int, width: int) -> np.ndarray:
    return np.zeros((height // KERNEL_SIZE, width // KERNEL_SIZE), dtype=bool)


def get_top_patches(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    sorted_patches = get_sorted_patches(image=image,
                                        height=height,
                                        width=width, )

    grid = get_grid(height=height, width=width)

    return get_selected_patches(sorted_patches=sorted_patches,
                                grid=grid,
                                height=height,
                                width=width, )


def get_shoelace_formula_result(x_coordinates: np.ndarray, y_coordinates: np.ndarray) -> float:
    forward_diagonal_product = x_coordinates * np.roll(y_coordinates, -1)
    backward_diagonal_product = y_coordinates * np.roll(x_coordinates, -1)

    return 0.5 * abs(np.sum(forward_diagonal_product - backward_diagonal_product))


def get_area_of_quadrilateral(points: np.ndarray) -> float:
    height_coordinates = points[:, 0]
    width_coordinates = points[:, 1]

    return get_shoelace_formula_result(height_coordinates, width_coordinates)


def draw_and_save(image: np.ndarray, points: np.ndarray, output_path: str) -> None:
    image_copy = image.copy()
    image_colored = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)

    points = points.reshape((1, -1, 1, 2))

    cv2.polylines(image_colored, points, isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imwrite(output_path, image_colored)


@error_handler
def process_image(image_path: str, output_path: str) -> None:
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        print(f"Error: Could not read image from path {image_path}. Please check the path and try again.")
        return

    patches = get_top_patches(gray_image)
    print("Area of the quadrilateral:", get_area_of_quadrilateral(patches))
    draw_and_save(gray_image, patches, output_path)
