# 测试脚本
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import imageio.v3 as iio
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img_origin = iio.imread('./img/Lenna_gray.bmp')
    img_test = iio.imread('./out/Lenna_gray.bmp')
    img_stack = np.hstack((img_origin, img_test))
    psnr = peak_signal_noise_ratio(img_origin, img_test)
    print('PSNR of "Lenna_gray.bmp":', psnr)

    img_origin = iio.imread('./img/building.bmp')
    img_test = iio.imread('./out/building.bmp')
    psnr = peak_signal_noise_ratio(img_origin, img_test)
    print('PSNR of "building.bmp":', psnr)
    img_stack = np.hstack((img_stack, img_origin, img_test))
    plt.imshow(img_stack, cmap='gray')
    plt.title("Comparison between original and compressed images");plt.show()
    img_size_origin = os.stat('./img/Lenna_gray.bmp').st_size
    img_size_test = os.stat('./out/Lenna_gray.my').st_size
    print('压缩率：', img_size_origin / img_size_test)
