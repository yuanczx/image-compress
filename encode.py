from sys import argv
import numpy as np
import cv2
from utils import run_length_encode, quant_matrix

if __name__ == "__main__":
    if len(argv) < 3: exit('传入参数不足')
    (in_img_path, out_img_path) = argv[1:3]

    # 将数据类型转为int16 防止零偏置转换时溢出。
    img = np.array(cv2.imread(in_img_path, 0), dtype=np.int16) - 128
    h, w = img.shape  # 获取图像大小
    block_size = 8  # 块大小
    # Zigzag矩阵
    z_mat = np.array([
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63])

    result_blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # 将512*512的矩阵分割为 4096 * 8*8 的子矩阵
            block = img[i:i + block_size, j:j + block_size]
            # 对子矩阵进行DCT变换
            dct_block = cv2.dct(np.float32(block))
            # 量化
            quan_block = np.round(dct_block / quant_matrix)
            # 使用 zigzag ‘之’字形重排序
            zigzag_block = quan_block.flatten() \
                [[np.where(z_mat == i)[0][0] for i in range(64)]]
            # 游程编码
            encoded = run_length_encode(zigzag_block)
            result_blocks.append(encoded)

    # 将所有块的Zigzag排序后结果连接起来
    result = np.concatenate(result_blocks)
    # 写入文件 类型为 int8
    result.astype(np.int8).tofile(out_img_path)
    exit('压缩完成，文件保存至 ' + out_img_path)
