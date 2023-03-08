import numpy as np
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

# 游程编码


def run_length_encoding(arr):
    # 初始化结果列表和计数器
    result = []
    count = 0

    # 遍历数组中的元素
    for ele in arr:
        # 如果元素为零，增加计数器
        if ele == 0:
            count += 1
        # 如果元素不为零或到达数组末尾，将计数器和元素添加到结果列表，并重置计数器
        else:
            result.append(count)
            result.append(ele)
            count = 0

    # 返回结果列表 使用255 表示 EOB
    result.append(125)
    return result


def run_length_decode(arr):
    result = []
    count = 0

    for i in range(len(arr)):
        if(arr[i] == 125):
            result += [0]*(64-len(result))
        elif i % 2 == 0:
            count = arr[i]
        else:
            result += [0]*count
            result.append(arr[i])
    return np.array(result)


# zigzag逆变换
def inverse_zigzag(arr):
    rows, cols = (8, 8)
    out = np.zeros((rows, cols))

    index = 0
    for i in range(rows + cols - 1):
        if i % 2 == 1:
            for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                out[i-j][j] = arr[index]
                index += 1
        else:
            for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                out[i - j][j] = arr[index]
                index += 1
    return out
