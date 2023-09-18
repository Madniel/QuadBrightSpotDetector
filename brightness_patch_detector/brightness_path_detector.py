from pathlib import Path
from typing import List, Any

import cv2
import numpy as np
from numpy import floating, ndarray

KERNEL_SIZE = 5
NUMBER_PATCHES = 4


def get_average_brightness(image: np.ndarray, x: int, y: int) -> float:
    patch = image[y:y + KERNEL_SIZE, x:x + KERNEL_SIZE]
    return np.mean(patch)


def get_patches(image: np.ndarray, height: int, width: int) -> List:
    patches = []
    for y in range(height - KERNEL_SIZE):
        for x in range(width - KERNEL_SIZE):
            average_brightness = get_average_brightness(image, x, y)
            patches.append(((x + 2, y + 2), average_brightness))
    return patches


def get_sorted_patches(image: np.ndarray, height: int, width: int) -> List:
    return sorted(get_patches(image, height, width), key=lambda x: x[1], reverse=True)


def get_selected_patches(sorted_patches: List, grid: np.ndarray, height: int, width: int) -> np.ndarray:
    selected_patches = np.empty((0, 2), int)  # Initialize an empty numpy array
    for patch in sorted_patches:
        coord, _ = patch
        grid_x = coord[0] // KERNEL_SIZE
        grid_y = coord[1] // KERNEL_SIZE
        if not grid[grid_y, grid_x]:  # if the cell is not occupied
            selected_patches = np.vstack([selected_patches, np.array([coord])])  # Stack the new coordinate
            grid[max(grid_y - 1, 0):min(grid_y + 2, height // KERNEL_SIZE),
            max(grid_x - 1, 0):min(grid_x + 2, width // KERNEL_SIZE)] = True
            if selected_patches.shape[0] == NUMBER_PATCHES:
                break
    return selected_patches


def get_grid(height: int, width: int) -> np.ndarray:
    return np.zeros((height // KERNEL_SIZE, width // KERNEL_SIZE), dtype=bool)


def get_top_patches(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    sorted_patches = get_sorted_patches(image=image,
                                        height=height,
                                        width=width, )

    # Create a 2D array to check if a patch has been selected in the vicinity
    grid = get_grid(height=height, width=width)

    return get_selected_patches(sorted_patches=sorted_patches,
                                grid=grid,
                                height=height,
                                width=width, )


def calculate_area_of_quadrilateral(x: List, y: List) -> float:
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(3)) + x[3] * y[0] - x[0] * y[3])


def get_area_of_quadrilateral(points: np.ndarray) -> float:
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return calculate_area_of_quadrilateral(x, y)


def draw_and_save(image: np.ndarray, points: np.ndarray, output_path: str) -> None:
    img_copy = image.copy()
    img_colored = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

    points = points.reshape((-1, 1, 2))

    cv2.polylines(img_colored, [points], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imwrite(output_path, img_colored)


def process_image(image_path: Path, output_path: Path) -> None:
    image_name = image_path.name
    output_name = output_path.name
    gray_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    patches = get_top_patches(gray_image)
    print("Area of the quadrilateral:", get_area_of_quadrilateral(patches))
    draw_and_save(gray_image, patches, output_name)
