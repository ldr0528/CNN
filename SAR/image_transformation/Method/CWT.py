import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure, transform
from skimage.color import rgb2gray, gray2rgb
import pywt

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
    r'D:\dataset\F2\Feature\CWT_2\Test\Cargo',
    r'D:\dataset\F2\Feature\CWT_2\Test\Tanker',
    r'D:\dataset\F2\Feature\CWT_2\Test\Fishing',
    r'D:\dataset\F2\Feature\CWT_2\Train\Cargo',
    r'D:\dataset\F2\Feature\CWT_2\Train\Tanker',
    r'D:\dataset\F2\Feature\CWT_2\Train\Fishing',
]

# Define the common size to which you want to resize the images
common_size = (256, 256)  # Adjust the size as needed

# Loop through the input directories
for i, input_image_dir in enumerate(input_image_dirs):
    output_image_dir = output_image_dirs[i]

    # Get a list of all JPEG files in the input directory
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.png'))]

    # Loop through the image files
    for image_file in image_files:
        # Check if the output .npy file already exists
        output_feature_file = os.path.join(output_image_dir, 'hog_features_' + image_file + '.npy')
        if os.path.exists(output_feature_file):
            continue  # Skip to the next image if features already exist

        # 1. Read the SAR image
        image = plt.imread(os.path.join(input_image_dir, image_file))

        # Check if the image is grayscale (single-channel)
        if len(image.shape) < 3 or image.shape[2] == 1:
            # Convert grayscale image to RGB
            image = gray2rgb(image)

        # 2. Resize the image to the common size
        output_shape = (common_size[0], common_size[1], image.shape[2])
        image = transform.resize(image, output_shape)

        # 3. Perform Continuous Wavelet Transform (CWT) on the SAR image
        cwt_result, frequencies = pywt.cwt(image, scales=np.arange(1, 10), wavelet='morl')

        # 4. Select a specific scale from the CWT result (equivalent to time plane in STFT)
        scale_index = 3  # Replace with the desired scale index
        selected_scale = cwt_result[scale_index]

        # 5. Extract HOG features from the selected scale
        fd, hog_image = hog(selected_scale, orientations=32, pixels_per_cell=(18, 18),
                            cells_per_block=(4, 4), visualize=True, channel_axis=-1)

        # 6. Rescale HOG image intensities for better visualization
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1))
        fd_reshaped = fd.reshape((256, -1))

        # 7. Save the HOG features to a .npy file
        hog_feature_file = os.path.join(output_image_dir, 'hog_features_' + image_file + '.npy')
        np.save(hog_feature_file, fd_reshaped)
#这段代码的目的是提取图像的HOG特征，HOG特征是一种用于描述图像纹理和形状的特征，常用于目标检测和图像分类任务。提取的HOG特征图被保存为.npy文件，可以用于后续的机器学习任务。
        # 8. Display and save the HOG features image (as in your original code)
        # fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        # hog_plot = axes.imshow(hog_image_rescaled, cmap='jet')
        # axes.axis('off')
        # plt.tight_layout()
        # result_file = os.path.join(output_image_dir, 'hog_' + image_file)
        # plt.savefig(result_file)
        # plt.close()