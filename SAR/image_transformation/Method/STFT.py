import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from skimage.feature import hog
from skimage import exposure, transform
from skimage.color import rgb2gray, gray2rgb

# Define the input and output directories
input_image_dirs = [
    r'D:\dataset\F2\fusarship\Test\Cargo',
    r'D:\dataset\F2\fusarship\Test\Tanker',
    r'D:\dataset\F2\fusarship\Test\Fishing',
    r'D:\dataset\F2\fusarship\Train\Cargo',
    r'D:\dataset\F2\fusarship\Train\Tanker',
    r'D:\dataset\F2\fusarship\Train\Fishing',
]
output_image_dirs = [
    r'D:\dataset\F2\Feature\STFT_2\Test\Cargo',
    r'D:\dataset\F2\Feature\STFT_2\Test\Tanker',
    r'D:\dataset\F2\Feature\STFT_2\Test\Fishing',
    r'D:\dataset\F2\Feature\STFT_2\Train\Cargo',
    r'D:\dataset\F2\Feature\STFT_2\Train\Tanker',
    r'D:\dataset\F2\Feature\STFT_2\Train\Fishing',
]

# Define the common size to which you want to resize the images
common_size = (256, 256)  # Adjust the size as needed

# Loop through the input directories
for i, input_image_dir in enumerate(input_image_dirs):
    output_image_dir = output_image_dirs[i]

    # Get a list of all JPEG files in the input directory
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg','.png'))]

    # Loop through the image files
    for image_file in image_files:
        # Check if the output image already exists
        output_file = os.path.join(output_image_dir, 'hog_' + image_file)
        if os.path.exists(output_file):
            continue

        # 1. Read the SAR image
        image = plt.imread(os.path.join(input_image_dir, image_file))

        # Check if the image is grayscale (single-channel)
        if len(image.shape) < 3 or image.shape[2] == 1:
            # Convert grayscale image to RGB
            image = gray2rgb(image)

        # 2. Resize the image to the common size
        output_shape = (common_size[0], common_size[1], image.shape[2])
        image = transform.resize(image, output_shape)

        # 3. Perform Short-time Fourier Transform on the SAR image
        f, t, Zxx = stft(image, nperseg=3, noverlap=1)

        # 4. Select a specific time plane from the STFT result
        time_plane = np.abs(Zxx[:, :, -1])  # Replace <index> with the desired time plane index

        # 5. Extract HOG features from the selected time plane
        fd, hog_image = hog(
            time_plane,
            orientations=32,  # 方向数
            pixels_per_cell=(18, 18),  # 像素单元数
            cells_per_block=(4, 4),  # 块单元数
            visualize=True,
            channel_axis=-1
        )


        # 6. Rescale HOG image intensities for better visualization
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1))

        fd_reshaped = fd.reshape((256, -1))

        # 7. Save the HOG features to a file
        hog_feature_file = os.path.join(output_image_dir, 'hog_features_' + image_file + '.npy')
        np.save(hog_feature_file, fd_reshaped)

        #8. Display and save the HOG features image (as in your original code)
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        hog_plot = axes.imshow(hog_image_rescaled, cmap='jet')
        axes.axis('off')
        plt.tight_layout()
        result_file = os.path.join(output_image_dir, 'hog_' + image_file)
        plt.savefig(result_file)
        plt.close()

