import cv2
import numpy as np
import os
import random

# 指定图像文件所在的目录
directory = 'D:\\dataset\\F2\\fusarship\\Fishing'

# 扩充次数
random_times = 20

# 获取目录下的所有文件
image_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.jpg')]

for m in image_files:
    img = cv2.imread(m)
    for px in range(random_times):
        check_times = random.randint(1, 5)
        choice_times = random.sample([1, 2, 3, 4, 5], check_times)
        # 随机旋转 90 180 270
        if 1 in choice_times:
            height, width, ch = img.shape
            M = cv2.getRotationMatrix2D((width / 2, height / 2), random.choice([0, 90, 180, 270]), 1.0)
            img = cv2.warpAffine(img, M, (width, height))
        # 翻转
        if 2 in choice_times:
            img = cv2.flip(img, 1)
        # 翻转
        if 3 in choice_times:
            img = cv2.flip(img, 0)
        # 随机中心缩放
        if 4 in choice_times:
            height, width, ch = img.shape
            length = int((random.randrange(60, 85) / 100) * ((height + width) / 2))
            x = width / 2 - length / 2
            y = height / 2 - length / 2

            img = img[int(y):int(y + length), int(x):int(x + length)]
            img = cv2.resize(img, (height, width))
        # 随机偏移
        if 5 in choice_times:
            img_w, img_h, _ = img.shape
            max_len = max(img_w, img_h) + 10

            new_img_shape = random.randint(max_len, int(max_len * 1.2))
            image = np.zeros([new_img_shape, new_img_shape, 3], np.uint8)

            x_this_img = random.randint(10, new_img_shape - img_w)
            y_this_img = random.randint(10, new_img_shape - img_h)

            for i in range(img_w):
                for j in range(img_h):
                    image[x_this_img + i, y_this_img + j, :] = img[i, j, :]
            img = cv2.resize(image, (height, width))

        # 获取保存文件名（在原文件名的基础上加上--和扩充次数）
        save_file_name = m.replace('.jpg', '--' + str(px) + '.jpg')
        cv2.imwrite(save_file_name, img)
