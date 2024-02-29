import cv2
import numpy as np
from MSST_Y_new import MSST
from sstn import SSTN
from skimage.feature import hog
from skimage import exposure
import glob
import tqdm
import matplotlib.pyplot as plt
import os

# 定义输入路径和对应的输出路径
input_output_paths = {
    r"D:\HuaweiMoveData\Users\Ldr13\Desktop\SAR\feature ext\example": {"MSST_output": r"D:\HuaweiMoveData\Users\Ldr13\Desktop\SAR\feature ext\example",
                                           "SSTN_output": r"D:\HuaweiMoveData\Users\Ldr13\Desktop\SAR\feature ext\result"},
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

        # Visualize and save images after SSTN transformation
        sstn_image = np.abs(sr_SST)

        plt.figure(figsize=(6, 6))
        plt.imshow(sstn_image, cmap='jet')
        #plt.title('Image after SSTN Transformation (Color)')
        plt.axis('off')

        # Save the image as EPS format
        plt.savefig(f'image_{m_index}_SSTN.eps', format='eps')

        # Apply HOG on SSTN transformed image
        fd_SSTN_hog, hog_image_SSTN = hog(sstn_image, orientations=32, pixels_per_cell=(18, 18), cells_per_block=(4, 4), visualize=True)

        # Rescale HOG image intensities for better visualization
        hog_image_SSTN_rescaled = exposure.rescale_intensity(hog_image_SSTN, in_range=(0, 1))

        # Display and save the HOG features image (as in your original code)
        plt.figure(figsize=(6, 6))
        plt.imshow(hog_image_SSTN_rescaled, cmap='jet')
        #plt.title('HOG Features after SSTN Transformation')
        plt.axis('off')

        # Save the image as EPS format
        plt.savefig(f'image_{m_index}_SSTN_HOG.eps', format='eps')

        plt.show()

