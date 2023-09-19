from pathlib import Path
from typing import List

import cv2
import numpy as np

KERNEL_SIZE = 5
NUMBER_PATCHES = 4
PATCH_CENTER_OFFSET = KERNEL_SIZE // 2


def get_average_brightness(image: np.ndarray, x: int, y: int) -> float:
    return np.mean(image[y:y + KERNEL_SIZE, x:x + KERNEL_SIZE]).astype(np.float32)


def get_patches(image: np.ndarray, height: int, width: int) -> List:
    patches = [((x + PATCH_CENTER_OFFSET, y + PATCH_CENTER_OFFSET), get_average_brightness(image, x, y))
               for x in range(width - KERNEL_SIZE)
               for y in range(height - KERNEL_SIZE)]
    return patches


def get_sorted_patches(image: np.ndarray, height: int, width: int) -> List:
    return sorted(get_patches(image, height, width), key=lambda x: x[1], reverse=True)


def get_selected_patches(sorted_patches: List, grid: np.ndarray, height: int, width: int) -> np.ndarray:
    selected_coordinates = []  # Use a temporary list for appending

    for coord, _ in sorted_patches:  # Direct unpacking in the loop header for clarity
        grid_x = coord[0] // KERNEL_SIZE
        grid_y = coord[1] // KERNEL_SIZE

        if not grid[grid_y, grid_x]:  # if the cell is not occupied
            selected_coordinates.append(coord)

            y_start, y_end = max(grid_y - 1, 0), min(grid_y + PATCH_CENTER_OFFSET, height // KERNEL_SIZE)
            x_start, x_end = max(grid_x - 1, 0), min(grid_x + PATCH_CENTER_OFFSET, width // KERNEL_SIZE)
            grid[y_start:y_end, x_start:x_end] = True

            if len(selected_coordinates) == NUMBER_PATCHES:
                break

    return np.array(selected_coordinates, dtype=int)


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


def process_image(image_path: Path, output_path: Path) -> None:
    image_name = image_path.name
    output_name = output_path.name
    gray_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    patches = get_top_patches(gray_image)
    print("Area of the quadrilateral:", get_area_of_quadrilateral(patches))
    draw_and_save(gray_image, patches, output_name)
