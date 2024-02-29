import cv2
import numpy as np
from MSST_Y_new import MSST
from sstn import SSTN
from skimage.feature import hog
import glob
import tqdm
from sklearn.decomposition import PCA
import joblib
import os

# 定义输入路径和对应的输出路径
input_output_paths = {
    r"D:\dataset\1\OSSHIP1\Train\Fishing": {"MSST_output": r"D:\dataset\1\Feature1\MSST_PCA1\Test\Fishing",
                                           "SSTN_output": r"D:\dataset\1\Feature1\SSTN_PCA1\Test\Fishing"},
}

# 初始化列表以存储所有图像的特征
all_hog_features_MSST = []
all_hog_features_SSTN = []

# 遍历每个输入文件夹
for input_path, output_paths in input_output_paths.items():
    image_data = glob.glob(input_path + '\\*')
    for m_index in tqdm.tqdm(range(len(image_data))):
        file_name = os.path.split(image_data[m_index])[-1].split('.')[0]
        msst_file_path = os.path.join(output_paths["MSST_output"], file_name + '-MSST.npy')
        sstn_file_path = os.path.join(output_paths["SSTN_output"], file_name + '-SSTN.npy')

        if os.path.exists(sstn_file_path):
            continue

        Data = cv2.imread(image_data[m_index], 0)
        Data = cv2.resize(Data, [256, 256]) / 256
        Row, Col = Data.shape
        sr_MSST = np.zeros([256, 256], dtype=complex)
        sr_SST = np.zeros([256, 256], dtype=complex)

        for i in range(Row):
            Ts = MSST(Data[i, :], 64, 1)
            sr_MSST[i, :] = sum(Ts)
            SST = SSTN(Data[i, :], 0.0100, 0.055)
            sr_SST[i, :] = sum(SST)

        time_plane_MSST = np.abs(sr_MSST)
        fd_MSST, _ = hog(time_plane_MSST, orientations=32, pixels_per_cell=(18, 18), cells_per_block=(4, 4), visualize=True)
        fd_MSST_reshaped = fd_MSST.reshape((256, -1))

        time_plane_SSTN = np.abs(sr_SST)
        fd_SSTN, _ = hog(time_plane_SSTN, orientations=32, pixels_per_cell=(18, 18), cells_per_block=(4, 4), visualize=True)
        fd_SSTN_reshaped = fd_SSTN.reshape((256, -1))

        all_hog_features_MSST.append(fd_MSST)
        all_hog_features_SSTN.append(fd_SSTN)

# 应用PCA到HOG特征矩阵
n_components = 100
pca_MSST = PCA(n_components=n_components)
pca_SSTN = PCA(n_components=n_components)
hog_feature_matrix_reduced_MSST = pca_MSST.fit_transform(np.vstack(all_hog_features_MSST))
hog_feature_matrix_reduced_SSTN = pca_SSTN.fit_transform(np.vstack(all_hog_features_SSTN))

# 保存PCA模型
# joblib.dump(pca_MSST, 'path/to/save/pca_model_MSST.pkl')
# joblib.dump(pca_SSTN, 'path/to/save/pca_model_SSTN.pkl')

# 将降维后的HOG特征保存到对应的输出文件夹
current_index_MSST = 0
current_index_SSTN = 0

for input_path, output_paths in input_output_paths.items():
    image_data = glob.glob(input_path + '\\*')
    for m_index in range(len(image_data)):
        file_name = os.path.split(image_data[m_index])[-1].split('.')[0]
        msst_file_path = os.path.join(output_paths["MSST_output"], file_name + '-MSST_reduced.npy')
        sstn_file_path = os.path.join(output_paths["SSTN_output"], file_name + '-SSTN_reduced.npy')

        # 保存MSST降维后的HOG特征
        np.save(msst_file_path, hog_feature_matrix_reduced_MSST[current_index_MSST])
        current_index_MSST += 1

        # 保存SSTN降维后的HOG特征
        np.save(sstn_file_path, hog_feature_matrix_reduced_SSTN[current_index_SSTN])
        current_index_SSTN += 1
