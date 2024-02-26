# 📷 QuadBrightSpotDetector

### Task Description

Write a Python script that:

- Reads an image from a file in grayscale.
- Finds the four non-overlapping 5x5 patches with the highest average brightness.
- Marks the patch centers as corners of a quadrilateral.
- Calculates its area in pixels.
- Draws the quadrilateral in red on the image.
- Saves the result in PNG format.

_Note: The task specifies the use of the opencv-python package for image handling and requires writing test cases._

---

### Repository Structure

📁 **brightness_patch_detector**: The main library responsible for processing the image according to the task's requirements.

📁 **scripts**: Contains the script `run_brightness_patch_detector`, facilitating task execution via the command line.

📁 **tests**: Hosts integral tests, unit tests, and the required resources for these tests.

📁 **utils**: Contains utilities like the error handling decorator.

---

## How It Works:

**1. Loading the Image:**

- The image is read in grayscale using OpenCV.
- The script ensures successful image loading and provides feedback if it encounters issues.

**2. Extracting and Ranking 5x5 Patches:**

- The image is dissected into 5x5 patches.
- Each patch's brightness is computed and subsequently ranked.

**3. Selecting Top Patches:**

- The brightest patches are prioritized.
- A grid mechanism ensures the patches don't overlap.

**4. Quadrilateral Formation:**

- The geometric centers of the patches are identified.
- These centers form the quadrilateral's vertices, which are sorted counterclockwise.

**5. Area Calculation:**

- The Shoelace formula aids in determining the quadrilateral's area.

**6. Drawing and Saving:**

- The quadrilateral is accentuated in red on the image.
- The modified image is saved in the PNG format.

**7. Exception Handling:**

- The `error_handler` decorator is designed to capture exceptions and offer user-friendly feedback.

---

## Usage:

Execute the script via the command line, specifying paths for both the input and resulting images.

**Command format**:

```bash
#!/usr/bin/python3
python brightness_patch_detector.py <path_to_image> <output_image_path> 
```

- **path_to_image**: The path to your grayscale image that you want to process.
- **output_image_path**: The path where you'd like to save the resulting image. If this is not provided, the default is
  **output.png**.

## 🛠 **Unit Tests**

Unit tests focus on individual components, ensuring their proper functionality. Our suite covers:

1. **Average Brightness**: Gauging average patch brightness.
2. **Patch Insights**: Extracting center locations & brightness values.
3. **Brightness Sorting**: Arranging patches by brightness level.
4. **Centroid Mechanics**: Accurate centroid computations.
5. **Centroid-Based Sorting**: Points sorted counter-clockwise around centroids.
6. **Patch Selection**: Filtering patches by brightness & position.
7. **Grid Dynamics**: Grid formulation based on image dimensions.
8. **Brightness Elites**: Identifying top patches.
9. **Area Analysis**: Quadrilateral and shoelace formula-based calculations.
10. **Final Visualization**: Drawing quadrilaterals & saving outputs.

---

## 🌐 **Integral Test**

A test, simulating real-world usage:
- **End-to-End Image Processing**: Processes images, detects brightness patches, and verifies outputs against established ground truths.

---
