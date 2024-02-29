import os
import shutil

# 源文件夹和目标文件夹路径
source_folder = "D:\\dataset\\Feature\\MSST+HOG_1\\Test\\Tanker"
target_folder = "D:\\dataset\\Feature\\SSTN+HOG_1\\Test\\Tanker"

# 需要匹配的文件名部分
matching_part = "SSTN.npy"

# 检查文件夹是否存在
if not os.path.exists(source_folder):
    print(f"源文件夹 {source_folder} 不存在。")
elif not os.path.exists(target_folder):
    print(f"目标文件夹 {target_folder} 不存在。")
else:
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件名是否包含指定的部分
        if matching_part in filename:
            source_file = os.path.join(source_folder, filename)
            target_file = os.path.join(target_folder, filename)

            # 如果目标文件夹中存在同名文件，则覆盖
            if os.path.exists(target_file):
                os.remove(target_file)

            # 移动文件
            shutil.move(source_file, target_file)
            #shutil.copy2(source_file, target_file)

    print("文件移动完成。")


