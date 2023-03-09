from sys import argv
import numpy as np
import cv2

from utils import inverse_zigzag, run_length_decode,quant_matrix

def image_decode(img_path):
    with open(img_path, 'rb') as f:
        img_bin = np.fromfile(f, dtype=np.int8)

    img_blocks = np.split(img_bin,np.where(img_bin==125)[0]+1)

    result = []
    for block in img_blocks:
        if(len(block) == 0): break
        im = run_length_decode(block) #对游程编码进行解码
        im = inverse_zigzag(im) # 反ziazag排序
        im = im * quant_matrix # 反量化
        im = cv2.idct(im) # 逆DCT变换
        result.append(im)

    # 遍历一维数组，将每个8*8的子矩阵复制到对应位置的512*512矩阵中
    decoded = np.zeros((512, 512))
    for i in range(4096):
        row = i // 64
        col = i % 64
        decoded[row*8:(row+1)*8, col*8:(col+1)*8] = np.reshape(result[i], (8, 8))
    return decoded


if __name__ == "__main__":
    
    in_image = ''
    out_image = ''

    try:
        in_image = argv[1]
        out_image = argv[2]
    except Exception as e:
        exit('传入参数错误')
    img = image_decode(in_image)
    cv2.imwrite(out_image,img)


    
    
    
    



