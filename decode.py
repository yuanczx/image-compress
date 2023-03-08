from sys import argv
import numpy as np
import cv2

from utils import inverse_zigzag, run_length_decode,quant_matrix

if __name__ == "__main__":
    
    in_image = ''
    out_image = ''

    try:
        in_image = argv[1]
        out_image = argv[2]
    except Exception as e:
        exit('传入参数错误')

    img_bin = np.zeros([],dtype=np.int8)
    try:
        with open(in_image,'rb') as f:
            # 读取图片数据
            img_bin = np.fromfile(f,dtype=np.int8)
    except Exception as e:
            exit('读取文件错误')
    # 恢复原始数据
    img = inverse_zigzag(run_length_decode(img_bin))
    
    
    



