import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from skimage.feature import hog
from skimage import exposure, transform
from skimage.color import rgb2gray, gray2rgb
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
    r'D:\dataset\1\Feature1\STFT_PCA1\Test\Fishing',
    r'D:\dataset\1\Feature1\STFT_PCA1\Test\Tanker',
    r'D:\dataset\1\Feature1\STFT_PCA1\Test\Passenger',
    r'D:\dataset\1\Feature1\STFT_PCA1\Train\Fishing',
    r'D:\dataset\1\Feature1\STFT_PCA1\Train\Tanker',
    r'D:\dataset\1\Feature1\STFT_PCA1\Train\Passenger',
]

# Define the common size to which you want to resize the images
common_size = (256, 256)  # Adjust the size as needed

# Loop through the input directories
all_hog_features = []

# 遍历每个输入文件夹
for input_image_dir in input_image_dirs:
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:

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

        all_hog_features.append(fd)

# 将HOG特征列表转换为NumPy数组
hog_feature_matrix = np.vstack(all_hog_features)

# 应用PCA到HOG特征矩阵
n_components = 100  # 您可以根据需要调整这个值
pca = PCA(n_components=n_components)
hog_feature_matrix_reduced = pca.fit_transform(hog_feature_matrix)


# 将降维后的HOG特征保存到对应的输出文件夹
current_index = 0
for output_image_dir in output_image_dirs:
    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    # 获取对应输入目录中的图像文件列表
    input_image_dir = input_image_dirs[output_image_dirs.index(output_image_dir)]
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.png'))]

    # 为每个图像文件保存对应的降维特征
    for image_file in image_files:
        reduced_feature_file = os.path.join(output_image_dir, 'reduced_hog_' + os.path.splitext(image_file)[0] + '.npy')
        np.save(reduced_feature_file, hog_feature_matrix_reduced[current_index])
        current_index += 1

    # 检查索引是否超出范围
    if current_index >= len(hog_feature_matrix_reduced):
        break

