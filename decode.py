from sys import argv
import numpy as np
import cv2
from utils import inverse_zigzag, run_length_decode, quant_matrix, EOB


def image_decode(img_data):
    # 将一维数组按照块结束标记EOB 分割为 8*8的小块
    img_blocks = np.split(img_data, np.where(img_data == EOB)[0] + 1)

    result = []
    for block in img_blocks:
        if len(block) == 0: break
        block_img = run_length_decode(block)  # 对游程编码进行解码
        block_img = inverse_zigzag(block_img)  # 反zigzag排序
        block_img = block_img * quant_matrix  # 反量化
        block_img = cv2.idct(block_img)  # 逆DCT变换
        result.append(block_img)

    # 遍历一维数组，将每个8*8的子矩阵复制到对应位置的512*512矩阵中
    decoded = np.zeros((512, 512))
    for i in range(4096):
        row = i // 64
        col = i % 64
        decoded[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] \
            = np.reshape(result[i], (8, 8))
    return decoded


if __name__ == "__main__":
    if len(argv) < 3: exit('传入参数错误')
    (in_img_path, out_img_path) = argv[1:3]
    with open(in_img_path, 'rb') as f:
        img_bin = np.fromfile(f, np.int8)
    img = image_decode(img_bin) + 128  # 零偏置转换
    cv2.imwrite(out_img_path, img)
    exit('解压完成，图像保存至 ' + out_img_path)
