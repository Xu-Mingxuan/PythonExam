import os
from tkinter import Image

import numpy as np

# 文件夹路径
folder_path = 'data_cat_dog'

# 初始化累积变量
total_pixels = 0
sum_normalized_pixel_value = np.zeros(3)   # 如果是RGM图像，需要三个通道的均值和方差


# 遍历文件夹中的图片文件
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg','.jpeg','.png','.bmp')):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            image_arry = np.array(image)


            #归一化像素值来到0-1之间
            normalized_image_arry = image_arry/255.0

            total_pixels += normalized_image_arry.size
            sum_normalized_pixel_value += np.sum(normalized_image_arry, axis=(0, 1))

# 计算均值和方差
mean = sum_normalized_pixel_value/ total_pixels

sum_squared_diff = np.zeros(3)
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg','.jpeg','.png','.bmp')):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            image_arry = np.array(image)
            # 归一化到0-1之间
            normalized_image_arry = image_arry/255.0
            print(normalized_image_arry.shape)
            print(mean.shape)
            print(image_path)

            try:
                diff = (normalized_image_arry - mean)**2
                sum_squared_diff += np.sum(diff, axis=(0, 1))
            except:
                print("捕获到自定义异常")

variance = sum_squared_diff/total_pixels

print("mean: ",mean)
print("variance: ",variance)



