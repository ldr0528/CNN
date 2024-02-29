import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
from skimage.color import rgb2gray

# Define input folder paths
input_folder_paths = [
    r'D:\dataset\F2\fusarship\Test\Cargo',
    r'D:\dataset\F2\fusarship\Test\Tanker',
    r'D:\dataset\F2\fusarship\Test\Fishing',
    r'D:\dataset\F2\fusarship\Train\Cargo',
    r'D:\dataset\F2\fusarship\Train\Tanker',
    r'D:\dataset\F2\fusarship\Train\Fishing',
]

# Define target image size
target_size = (256, 256)  # For example, set as 256x256

# Define output folder paths
output_folder_paths = [
    r'D:\dataset\F2\Feature\HOG_2\Test\Cargo',
    r'D:\dataset\F2\Feature\HOG_2\Test\Tanker',
    r'D:\dataset\F2\Feature\HOG_2\Test\Fishing',
    r'D:\dataset\F2\Feature\HOG_2\Train\Cargo',
    r'D:\dataset\F2\Feature\HOG_2\Train\Tanker',
    r'D:\dataset\F2\Feature\HOG_2\Train\Fishing',
]

# Loop over each input folder path
for input_folder_path, output_folder_path in zip(input_folder_paths, output_folder_paths):
    # Get all image files in the input folder
    image_files = [f for f in os.listdir(input_folder_path) if f.endswith(('.jpg', '.png'))]

    # Loop over each image file
    for image_file in image_files:
        # Check if the output image and HOG feature file already exist
        result_image_file = os.path.join(output_folder_path, 'hog_' + image_file)
        result_feature_file = os.path.join(output_folder_path, 'hog_features_' + image_file + '.npy')
        if os.path.exists(result_image_file):
            continue

        # 1. Read the SAR image and resize it
        image = cv2.imread(os.path.join(input_folder_path, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, target_size)

        # 2. Convert the image to grayscale
        image_gray = rgb2gray(image_resized)

        # 3. Extract HOG features from the grayscale image
        fd, hog_image = hog(image_gray, orientations=32, pixels_per_cell=(18, 18),
                            cells_per_block=(4, 4), visualize=True)

        # 4. Rescale the HOG image intensity for better visualization
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1))
        fd_reshaped = fd.reshape((256, -1))

        # Save the HOG features as a NumPy array
        np.save(result_feature_file,fd_reshaped)