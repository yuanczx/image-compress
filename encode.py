from sys import argv
import numpy as np
import cv2

from utils import run_length_encoding,quant_matrix

if __name__ == "__main__":
    if len(argv)<3 : exit('传入参数不足')
    in_img_path = argv[1]
    out_img_path = argv[2]
    img = cv2.imread(in_img_path,0)
    # 获取图像大小
    h, w = img.shape
    # 块大小
    block_size = 8

    # Zigzag矩阵
    zigzag_matrix = np.array([
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    ])

    # 将图像划分成8x8块
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            blocks.append(block)

    # 对每个块进行离散余弦变换
    dct_blocks = []
    for block in blocks:
        dct_block = cv2.dct(np.float32(block))
        dct_blocks.append(dct_block)
    
    result_blocks = []
    for dct_block in dct_blocks:
        # 量化
        quan_block = np.round(dct_block / quant_matrix)
        # 使用 zigzag ‘之’字形重排序
        zigzag_block = quan_block.flatten()[[np.where(zigzag_matrix == index)[0][0] for index in range(64)]]
        encoded = run_length_encoding(zigzag_block)
        # zigzag_dict = dict(zip(zigzag_matrix,quan_block.flatten()))
        # zigzag_block = []
        # for key in sorted(zigzag_dict):
        #     zigzag_block.append(zigzag_dict[key])
        result_blocks.append(encoded)

    # 将所有块的Zigzag排序后结果连接起来
    result = np.concatenate(result_blocks)
    # 写入文件 类型为 int8
    result.astype(np.int8).tofile(out_img_path)
    exit('压缩完成，文件保存至 '+out_img_path)

