from sys import argv
import numpy as np
import cv2

from utils import inverse_zigzag, run_length_decode,quant_matrix

def image_decode(img_bin):

    # 将一维数组按照块结束标记 125 分割为 8*8的小块
    img_blocks = np.split(img_bin,np.where(img_bin==125)[0]+1)

    result = []
    for block in img_blocks:
        if(len(block) == 0): break
        img = run_length_decode(block) #对游程编码进行解码
        img = inverse_zigzag(img) # 反ziazag排序
        img = img * quant_matrix # 反量化
        img = cv2.idct(img) # 逆DCT变换
        result.append(img)

    # 遍历一维数组，将每个8*8的子矩阵复制到对应位置的512*512矩阵中
    decoded = np.zeros((512, 512))
    for i in range(4096):
        row = i // 64
        col = i % 64
        decoded[row*8:(row+1)*8, col*8:(col+1)*8] = np.reshape(result[i], (8, 8))
    return decoded


if __name__ == "__main__":
    
    if len(argv)<3: exit('传入参数错误')
    in_img_path = argv[1]
    out_img_path = argv[2]
    with open(in_img_path,'rb') as f:
        img_bin = np.fromfile(f,np.int8)
    img = image_decode(img_bin)
    cv2.imwrite(out_img_path,img)


    
    
    
    



