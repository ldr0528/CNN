import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
import joblib

# Define input folder paths
input_folder_paths = [
    r'D:\dataset\1\OSSHIP1\Test\Fishing',
    r'D:\dataset\1\OSSHIP1\Test\Tanker',
    r'D:\dataset\1\OSSHIP1\Test\Passenger',
    r'D:\dataset\1\OSSHIP1\Train\Fishing',
    r'D:\dataset\1\OSSHIP1\Train\Tanker',
    r'D:\dataset\1\OSSHIP1\Train\Passenger',
]

# Define target image size
target_size = (256, 256)  # For example, set as 256x256

# Define output folder paths
output_folder_paths = [
    r'D:\dataset\1\Feature1\HOG_PCA1\Test\Fishing',
    r'D:\dataset\1\Feature1\HOG_PCA1\Test\Tanker',
    r'D:\dataset\1\Feature1\HOG_PCA1\Test\Passenger',
    r'D:\dataset\1\Feature1\HOG_PCA1\Train\Fishing',
    r'D:\dataset\1\Feature1\HOG_PCA1\Train\Tanker',
    r'D:\dataset\1\Feature1\HOG_PCA1\Train\Passenger',
]

# 初始化一个列表以存储所有图像的HOG特征
all_hog_features = []

# 遍历每个输入文件夹路径
for input_folder_path, output_folder_path in zip(input_folder_paths, output_folder_paths):
    image_files = [f for f in os.listdir(input_folder_path) if f.endswith(('.jpg', '.png'))]
    for image_file in image_files:
        image_path = os.path.join(input_folder_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, target_size)
        image_gray = rgb2gray(image_resized)

        # 提取HOG特征
        fd, hog_image = hog(image_gray, orientations=32, pixels_per_cell=(18, 18),
                            cells_per_block=(4, 4), visualize=True)

        # 将HOG特征添加到列表中
        all_hog_features.append(fd)

        # 保存原始HOG特征到对应的输出文件夹
        # original_feature_file = os.path.join(output_folder_path, 'original_hog_' + image_file + '.npy')
        # np.save(original_feature_file, fd)

# 将HOG特征列表转换为NumPy数组
hog_feature_matrix = np.vstack(all_hog_features)

# 应用PCA到HOG特征矩阵
n_components = 100
pca = PCA(n_components=n_components)
hog_feature_matrix_reduced = pca.fit_transform(hog_feature_matrix)

# 保存PCA模型
#joblib.dump(pca, r'D:\dataset\Feature\HOG_2\pca_model.pkl')

# 将降维后的HOG特征保存到对应的输出文件夹
current_index = 0

for i, (input_folder_path, output_folder_path) in enumerate(zip(input_folder_paths, output_folder_paths)):
    image_files = [f for f in os.listdir(input_folder_path) if f.endswith(('.jpg', '.png'))]
    for j, image_file in enumerate(image_files):
        reduced_feature_file = os.path.join(output_folder_path, 'reduced_hog_' + image_file + '.npy')

        # 保存当前图像的降维HOG特征
        np.save(reduced_feature_file, hog_feature_matrix_reduced[current_index])

        # 更新索引
        current_index += 1
