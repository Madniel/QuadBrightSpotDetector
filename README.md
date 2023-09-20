# amphiprion_task
Repository contains coding challenge from Amphiprion company.
####
Write a Python script that reads an image from a file as grayscale, and finds the four non-overlapping 5x5 patches with highest average brightness. Take the patch centers as corners of a quadrilateral, calculate its area in pixels, and draw the quadrilateral in red into the image and save it in PNG format. Use the opencv-python package for image handling. Write test cases.  
####  
Folder contains:
- Library brightness_patch_detector responsible for processing image for following task
- scripts contain script ***run_brightness_patch_detector*** responsible for running task in command line. Script require two arguments: image_path and output_path. \
  Exemplary command: ***python brightness_patch_detector.py <path_to_image> <output_image_path>***
- tests contains integral test, unit tests and resources to tests
- utils contains decorator for error handling
