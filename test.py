import numpy as np
import cv2
from utils import run_length_decode, run_length_encoding,inverse_zigzag
def test(img):
    # 获取图像大小
    h, w = img.shape[:2]
    # 块大小
    block_size = 8
    # 量化矩阵
    quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

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
    
    print(len(dct_blocks))
    result_blocks = []
    for dct_block in dct_blocks:
        # 量化
        quan_block = np.round(dct_block / quant_matrix)
        # zig-zag ‘之’字形重排序
        # print(quan_block)
        zigzag_block = quan_block.flatten()[[np.where(zigzag_matrix == index)[0][0] for index in range(64)]]
        encoded = run_length_encoding(zigzag_block)
        # zigzag_dict = dict(zip(zigzag_matrix,quan_block.flatten()))
        # zigzag_block = []
        # for key in sorted(zigzag_dict):
        #     zigzag_block.append(zigzag_dict[key])
        result_blocks.append(encoded)
    # 将所有块的Zigzag扫描结果连接起来
    result = np.concatenate(result_blocks)
    # print(result.shape)
    return result


a = np.array([-76, -73, -67, -62, -58, -67, -64, -55,
              -65, -69, -62, -38, -19, -43, -59, -56,
              -66, -69, -60, -15, 16, -24, -62, -55,
              -65, -70, -57, -6, 26, -22, -58, -59,
              -61, -67, -60, -24, -2, -40, -60, -58,
              -49, -63, -68, -58, -51, -65, -70, -53,
              -43, -57, -64, -69, -73, -67, -63, -45,
              -41, -49, -59, -60, -63, -52, -50, -34]).reshape(8, 8)
b = np.array([-26, -3, 1, -3, -2,  -6,   2,  -4,
              1,  -4,   1,   1,   5,   0,   2,   0,
              0,  -1, 2, 0, 0, 0, 0, 0,
              -1, -1, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0,   0,   0,   0,
              0,   0,   0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,   0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0])

# # r = test(a)
r = test(cv2.imread('./img/Lenna_gray.bmp',0))
# r.astype(np.int8).tofile('./test.my')
# print(r)

# with open('./test.my', 'rb') as f:
#     img_bin = np.fromfile(f, dtype=np.int8)
#     # print(img_bin)
#     img_bins = np.split(img_bin,np.where(img_bin==125)[0]+1)
#     print(len(img_bins))
#     for value in img_bins:
#         if(len(value) == 0): break
#         img = run_length_decode(value)
#         print(img)
#         img = inverse_zigzag(img)
#         print(img)

