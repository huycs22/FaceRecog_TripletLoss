# USAGE
# python crop_faces.py --dataset train_dataset --output cropped_train_dataset
# python crop_faces.py --dataset test_dataset --output cropped_test_dataset

from imutils.paths import list_images
from tqdm import tqdm
import numpy as np
import argparse
import cv2
import os
from mtcnn import MTCNN

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='casia-webface',
    help="Path to input dataset")
ap.add_argument("-o", "--output", default='cropped_test_dataset',
    help="Path to output dataset")
args = vars(ap.parse_args())

# Initialize the MTCNN face detector
print("[INFO] loading MTCNN face detector...")
detector = MTCNN()

# Check if the output dataset directory exists; if not, create it
if not os.path.exists(args["output"]):
    os.makedirs(args["output"])

# Grab the file and sub-directory names in the dataset directory
print("[INFO] grabbing the names of files and directories...")
names = os.listdir(args["dataset"])

# Loop over all names
print("[INFO] starting to crop faces and saving them to disk...")
for name in tqdm(names):
    # Build directory path
    dirPath = os.path.join(args["dataset"], name)

    # Check if the directory path is a directory
    if os.path.isdir(dirPath):
        # Grab the path to all the images in the directory
        imagePaths = list(list_images(dirPath))

        # Build the path to the output directory
        outputDir = os.path.join(args["output"], name)

        # Check if the output directory exists; if not, create it
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        # Loop over all image paths
        for imagePath in imagePaths:
            # Grab the image ID, load the image, and grab the dimensions
            imageID = os.path.basename(imagePath)
            image = cv2.imread(imagePath)
            if image is None:
                continue  # Skip if the image is not loaded properly
            (h, w) = image.shape[:2]

            # Convert the image from BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            detections = detector.detect_faces(rgb_image)

            # Proceed if at least one face is detected
            if detections:
                # Select the detection with the highest confidence
                best_detection = max(detections, key=lambda det: det['confidence'])
                confidence = best_detection['confidence']

                # Define a confidence threshold (e.g., 0.5)
                if confidence >= 0.5:
                    x, y, width, height = best_detection['box']
                    x, y = max(0, x), max(0, y)
                    endX, endY = x + width, y + height

                    # Ensure the bounding box is within the image dimensions
                    endX = min(endX, w)
                    endY = min(endY, h)

                    # Extract the face from the image
                    face = image[y:endY, x:endX]

                    # Save the cropped face image to the output directory
                    facePath = os.path.join(outputDir, imageID)
                    cv2.imwrite(facePath, face)

print("[INFO] finished cropping faces and saving them to disk...")
