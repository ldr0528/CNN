# 导入所需的库
import numpy as np

# 读取.npy文件
npy_file = './reduced_hog_180_12003_0_0_vh_augmented_0_augmented_0.jpg.npy'  # 替换为你的.npy文件的路径
npy_data = np.load(npy_file)

# 打印读取的数据
print("读取的.npy文件数据:")
print(npy_data)
