import cv2
import numpy as np
from MSST_Y_new import MSST
from sstn import SSTN
from skimage.feature import hog
import glob
import tqdm    #tqdm模块是一个快速、可扩展的Python进度条工具库。
import os

# 定义输入路径和对应的输出路径
input_output_paths = {
    r"D:\dataset\F2\fusarship\Train\Tanker": {"MSST_output": r"D:\dataset\F2\Feature\MSST_2\Train\Tanker",
                                              "SSTN_output": r"D:\dataset\F2\Feature\SSTN_2\Train\Tanker"},
}

for input_path, output_paths in input_output_paths.items():
    image_data = glob.glob(input_path + '\\*')
    #每当这个循环执行一次迭代时，tqdm 将更新进度条。
    for m_index in tqdm.tqdm(range(len(image_data))):
        file_name = os.path.split(image_data[m_index])[-1].split('.')[0]  # 提取文件名，不包含扩展名
        msst_file_path = os.path.join(output_paths["MSST_output"], file_name + '-MSST.npy')
        sstn_file_path = os.path.join(output_paths["SSTN_output"], file_name + '-SSTN.npy')

        # 检查文件是否已存在，如果存在则跳过
        if  os.path.exists(sstn_file_path):
            continue

        Data = cv2.imread(image_data[m_index], 0)
        Data = cv2.resize(Data, [256, 256]) / 256
        Row, Col = Data.shape
        sr_MSST = np.zeros([256, 256], dtype=complex)
        sr_SST = np.zeros([256, 256], dtype=complex)

        for i in range(Row):
            #MSST
            Ts = MSST(Data[i, :], 64, 1)
            sr_MSST[i, :] = sum(Ts)

            # SSTN
            SST = SSTN(Data[i, :], 0.0100, 0.055)
            sr_SST[i, :] = sum(SST)

        # 提取 HOG 特征
        time_plane = np.abs(sr_MSST)
        fd, _ = hog(time_plane, orientations=32, pixels_per_cell=(18, 18), cells_per_block=(4, 4), visualize=True)
        fd_reshaped = fd.reshape((256, -1))

        time_plane1 = np.abs(sr_SST)
        fd1, _ = hog(time_plane1, orientations=32, pixels_per_cell=(18, 18), cells_per_block=(4, 4), visualize=True)
        fd1_reshaped = fd1.reshape((256, -1))

        # 保存特征
        np.save(msst_file_path, fd_reshaped)
        np.save(sstn_file_path, fd1_reshaped)