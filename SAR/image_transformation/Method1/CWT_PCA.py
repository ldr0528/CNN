import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure, transform
from skimage.color import rgb2gray, gray2rgb
import pywt
from sklearn.decomposition import PCA

# Define the input and output directories
input_image_dirs = [
    r'D:\dataset\1\OSSHIP1\Test\Fishing',
    r'D:\dataset\1\OSSHIP1\Test\Tanker',
    r'D:\dataset\1\OSSHIP1\Test\Passenger',
    r'D:\dataset\1\OSSHIP1\Train\Fishing',
    r'D:\dataset\1\OSSHIP1\Train\Tanker',
    r'D:\dataset\1\OSSHIP1\Train\Passenger',
]
output_image_dirs = [
    r'D:\dataset\1\Feature1\CWT_PCA1\Test\Fishing',
    r'D:\dataset\1\Feature1\CWT_PCA1\Test\Tanker',
    r'D:\dataset\1\Feature1\CWT_PCA1\Test\Passenger',
    r'D:\dataset\1\Feature1\CWT_PCA1\Train\Fishing',
    r'D:\dataset\1\Feature1\CWT_PCA1\Train\Tanker',
    r'D:\dataset\1\Feature1\CWT_PCA1\Train\Passenger',
]

# Define the common size to which you want to resize the images
common_size = (256, 256)  # Adjust the size as needed

# Loop through the input directories
for i, input_image_dir in enumerate(input_image_dirs):
    output_image_dir = output_image_dirs[i]

    # Get a list of all JPEG files in the input directory
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.png'))]



all_hog_features = []

# 循环通过输入目录
for i, input_image_dir in enumerate(input_image_dirs):
    output_image_dir = output_image_dirs[i]
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:
        output_feature_file = os.path.join(output_image_dir, 'hog_features_' + image_file + '.npy')
        if os.path.exists(output_feature_file):
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

        # 3. Perform Continuous Wavelet Transform (CWT) on the SAR image
        cwt_result, frequencies = pywt.cwt(image, scales=np.arange(1, 10), wavelet='morl')

        # 4. Select a specific scale from the CWT result (equivalent to time plane in STFT)
        scale_index = 3  # Replace with the desired scale index
        selected_scale = cwt_result[scale_index]

        # 5. Extract HOG features from the selected scale
        fd, hog_image = hog(selected_scale, orientations=32, pixels_per_cell=(18, 18),
                            cells_per_block=(4, 4), visualize=True, channel_axis=-1)

        # 保存HOG特征到列表中
        all_hog_features.append(fd)

# 将HOG特征列表转换为NumPy数组
hog_feature_matrix = np.vstack(all_hog_features)

# 应用PCA到HOG特征矩阵
n_components = 100  # 可以根据需要调整
pca = PCA(n_components=n_components)
hog_feature_matrix_reduced = pca.fit_transform(hog_feature_matrix)


# 将降维后的HOG特征保存到对应的输出文件夹
current_index = 0
for i, output_image_dir in enumerate(output_image_dirs):
    image_files = [f for f in os.listdir(input_image_dirs[i]) if f.endswith(('.jpg', '.png'))]
    for image_file in image_files:
        reduced_feature_file = os.path.join(output_image_dir, 'reduced_hog_features_' + image_file + '.npy')
        np.save(reduced_feature_file, hog_feature_matrix_reduced[current_index])
        current_index += 1