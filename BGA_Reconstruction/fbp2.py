import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import time
from scipy.fft import fft, ifft
from numpy.fft import fftshift

def projection(image, theta):
    """
    計算投影值
    :param image: 原始圖像
    :param theta: 射束旋轉角度
    :return: 投影值矩陣
    projectionNum: 射束個數
    thetaNum: 角度個數
    randontansform: 拉登變換結果
    """
    print('正在計算投影...')

    projectionNum = len(image[0])

    thetaNum = len(theta)
 
    radontansform = np.zeros((projectionNum, thetaNum), dtype='float64')
    for i in range(len(theta)):
        # 進行離散拉登變換
        rotation = ndimage.rotate(image, -theta[i], reshape=False).astype('float64')
     
        radontansform[:, i] = sum(rotation)
    return radontansform


def ramFilter(theta, size):

    sinogram = projectionValue.T
    # fouriner transform
    sin_fft = []
    for i in range(len(theta)):
        fft_row = fft(sinogram[i])
        sin_fft.append(fft_row)
    sin_fft = np.array(sin_fft)

    # filter
    fft_filter = []
    ram_filter = np.linspace(-1,1,size)
    ram_filter = abs(ram_filter)
    ram_filter = fftshift(ram_filter)

    for i in range(len(theta)):
        filter_row = np.multiply(sin_fft[i], ram_filter)
        fft_filter.append(filter_row)
    fft_filter = np.array(fft_filter)

    # fouriner inverse transform
    filetr_ifft = []
    for i in range(len(theta)):
        ifft_row = ifft(fft_filter[i])
        filetr_ifft.append(ifft_row.real)
    filtered_sinogram = np.array(filetr_ifft)
    filtered_sinogram_normalized = np.copy(filtered_sinogram)
    # cv2.normalize(filtered_sinogram, filtered_sinogram_normalized, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("new_fbp_sino.png", filtered_sinogram)

    return filtered_sinogram.T
    


def systeMartrix(theta, size, num, delta):
    """
    計算系統矩陣
    :param theta: 射束旋轉角度
    :param size: 圖片尺寸
    :param num: 射束條數
    :param delat: 網格邊長
    :return: gridNum:穿過網格編號，gridLen:穿過網格長度
    """
    print('正在計算系統矩陣...')

    start = time.perf_counter()
    functionNum = len(theta) * num  # 方程個數，即為系統矩陣的行數
    # 射線最多穿過2*size個方格
    gridNum = np.zeros((functionNum, 2 * size))  # 系統矩陣：編號
    gridLen = np.zeros((functionNum, 2 * size))  # 系統矩陣：長度
    N = np.arange(-(size - 1) / 2, (size - 1) / 2 + 1)  # 射束
    for loop1 in range(len(theta)):
        th = theta[loop1]  # 射束角度
        for loop2 in range(size):
            u = np.zeros((2 * size))  # 編號
            v = np.zeros((2 * size))  # 長度

            # 垂直入射
            if th == 0:
                # 射束未穿過圖像時
                if (N[loop2] >= size / 2 * delta) or (N[loop2] <= -size / 2 * delta):
                    continue
                # 入射網格編號
                kin = np.ceil(size / 2 + N[loop2] / delta)
                # 穿過網格編號
                kk = np.arange(kin, (kin + size * size), step=size)
                
                u[0:size] = kk
                v[0:size] = np.ones(size) * delta

            # 平行入射
            elif th == 90:
                if (N[loop2] >= size / 2 * delta) or (N[loop2] <= -size / 2 * delta):
                    continue
                # 出射網格編號
                kout = size * np.ceil(size / 2 - N[loop2] / delta)
                kk = np.arange(kout - size + 1, kout + 1)
                u[0:size] = kk
                v[0:size] = np.ones(size) * delta

            else:
                # phi為射束與x軸所夾銳角
                if th > 90:
                    phi = th - 90
                elif th < 90:
                    phi = 90 - th
                # 角度值換算為弧度制
                phi = phi * np.pi / 180
                # 截距
                b = N / np.cos(phi)
                # 斜率
                m = np.tan(phi)
                # 入射點縱坐標
                y1 = -(size / 2) * delta * m + b[loop2]
                # 出射點縱坐標
                y2 = (size / 2) * delta * m + b[loop2]

                # 射束未穿過圖像
                if (y1 < -size / 2 * delta and y2 < -size / 2 * delta) or (
                        y1 > size / 2 * delta and y2 > size / 2 * delta):
                    continue

                # 穿過a、b邊（左側和上側）
                if (y1 <= size / 2 * delta and y1 >= -size / 2 * delta and y2 > size / 2 * delta):
                    """
                    (xin,yin): 入射點坐標
                    (xout,yout): 出射點坐標
                    kin,kout: 入射格子標號，出射格子編號
                    d1: 入射格子左下角與入射射束距離
                    """
                    yin = y1
                    yout = size / 2 * delta
                    # xin = -size / 2 * delta
                    xout = (yout - b[loop2]) / m
                    kin = size * np.floor(size / 2 - yin / delta) + 1
                    kout = np.ceil(xout / delta) + size / 2
                    d1 = yin - np.floor(yin / delta) * delta

                # 穿過a、c邊（左側和右側）
                elif (
                        y1 <= size / 2 * delta and y1 >= -size / 2 * delta and y2 >= -size / 2 * delta and y2 < size / 2 * delta):
                    # xin = -size / 2 * delta
                    # xout = size / 2 * delta
                    yin = y1
                    yout = y2
                    kin = size * np.floor(size / 2 - yin / delta) + 1
                    kout = size * np.floor(size / 2 - yout / delta) + size
                    d1 = yin - np.floor(yin / delta) * delta

                # 穿過d、b邊（下側和上側）
                elif (y1 < - size / 2 * delta and y2 > size / 2 * delta):
                    yin = - size / 2 * delta
                    yout = size / 2 * delta
                    xin = (yin - b[loop2]) / m
                    xout = (yout - b[loop2]) / m
                    kin = size * (size - 1) + size / 2 + np.ceil(xin / delta)
                    kout = np.ceil(xout / delta) + size / 2
                    d1 = size / 2 * delta + np.floor(xin / delta) * delta * m + b[loop2]

                # 穿過d、c邊（下側和右側）
                elif (y1 < - size / 2 * delta and y2 >= -size / 2 * delta and y2 < size / 2 * delta):
                    yin = -size / 2 * delta
                    yout = y2
                    xin = (yin - b[loop2]) / m
                    # xout = size / 2 * delta
                    kin = size * (size - 1) + size / 2 + np.ceil(xin / delta)
                    kout = size * np.floor(size / 2 - yout / delta) + size
                    d1 = size / 2 * delta + np.floor(xin / delta) * delta * m + b[loop2]

                else:
                    continue

                # 計算穿過的格子編號和長度
                """
                k: 射線穿過的格子編號
                c: 穿過格子的序號
                d2: 穿過的格子的右側與該方格右下角頂點的距離
                """
                k = kin  # 入射的格子即為穿過的第一個格子
                c = 0  # c為穿過格子的序號
                d2 = d1 + m * delta  # 與方格右側交點

                # 當方格數在1~n^2內叠代計算
                while k >= 1 and k <= np.power(size, 2):
                    """
                    根據射線與方格的左右兩側的交點關系，來確定穿過方格的六種情況。
                    在每種情況中，存入穿過的方格編號，穿過方格的射線長度。
                    若該方格是最後一個穿過的方格，則停止叠代；若不是最後一個方格，則計算下一個穿過的方格的編號、左右邊與射線的交點。
                    """
                    if d1 >= 0 and d2 > delta:
                        u[c] = k  # 穿過方格的編號
                        v[c] = (delta - d1) * np.sqrt(np.power(m, 2) + 1) / m  # 穿過方格的射線長度
                        if k > size and k != kout:  # 若不是最後一個方格
                            k -= size  # 下一個方格編號
                            d1 -= delta  # 下一個方格左側交點
                            d2 = delta * m + d1  # 下一個方格右側交點
                        else:  # 若是最後一個方格則直接跳出循環
                            break

                    elif d1 >= 0 and d2 == delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1)
                        if k > size and k != kout:
                            k -= size + 1
                            d1 = 0
                            d2 = d1 + m * delta
                        else:
                            break

                    elif d1 >= 0 and d2 < delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1)
                        if k != kout:
                            k += 1
                            d1 = d2
                            d2 = d1 + m * delta
                        else:
                            break

                    elif d1 <= 0 and d2 >= 0 and d2 <= delta:
                        u[c] = k
                        v[c] = d2 * np.sqrt(np.power(m, 2) + 1) / m
                        if k != kout:
                            k += 1
                            d1 = d2
                            d2 = d1 + m * delta
                        else:
                            break

                    elif d1 <= 0 and d2 > delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1) / m
                        if k > size and k != kout:
                            k -= size
                            d1 -= delta
                            d2 = d1 + m * delta
                        else:
                            break

                    elif d1 <= 0 and d2 == delta:
                        u[c] = k
                        v[c] = d2 * np.sqrt(np.power(m, 2) + 1) / m
                        if k > size and k != kout:
                            k -= size + 1
                            d1 = 0
                            d2 = delta * m
                        else:
                            break

                    else:
                        print(d1, d2, "數據錯誤!")

                    c += 1

                # 當射線斜率為負數時，利用對稱性進行計算
                if th < 90:
                    u_temp = np.zeros(2 * size)
                    # 排除掉未穿過圖像的射束
                    if u.any() == 0:
                        continue
                    # 射束穿過的格子的編號
                    indexMTZero = np.where(u > 0)

                    # 利用對稱性求得穿過網格編號
                    for loop in range(len(u[indexMTZero])):
                        """計算穿過的方格編號，利用方格編號與邊長的取余的關系，得到對稱的射線穿過的方格編號。"""
                        r = np.mod(u[loop], size)
                        if r == 0:
                            u_temp[loop] = u[loop] - size + 1
                        else:
                            u_temp[loop] = u[loop] - 2 * r + size + 1
                    u = u_temp

            # 方格編號
            gridNum[loop1 * num + loop2, :] = u
            # 穿過方格的射線長度
            gridLen[loop1 * num + loop2, :] = v

    end = time.perf_counter()
    print('系統矩陣計算耗時:%s 秒。'%(end - start))
    return gridNum, gridLen


def iteration(theta, size, gridNum, gridLen, F, ite_num):
    """
    按照公式叠代重建
    :param theta: 旋轉角度
    :param size: 圖像邊長
    :param gridNum: 射線穿過方格編號
    :param gridLen: 射線穿過方格長度
    :param F: 重建後圖像
    :return: 重建後圖像
    """
    print('正在進行叠代...')

    start = time.perf_counter()
    c = 0  # 叠代計數
    while (c < ite_num):
        for loop1 in range(len(theta)):  # 在角度theta下
            for loop2 in range(size):  # 第loop2條射線
                u = gridNum[loop1 * size + loop2, :]
                v = gridLen[loop1 * size + loop2, :]
                if u.any() == 0:  # 若射線未穿過圖像，則直接計算下一條射線
                    continue
                # 本條射線對應的行向量
                w = np.zeros(sizeSquare, dtype=np.float64)
              
                # 本條射線穿過的網格編號
                uLargerThanZero = np.where(u > 0)
                # 本條射線穿過的網格長度
                w[u[uLargerThanZero].astype(np.int64) - 1] = v[uLargerThanZero]
                # 計算估計投影值
                PP = w.dot(F)
                
                # 計算實際投影與估計投影的誤差
                error = projectionValueFiltered[loop2, loop1] 
                # error = projectionValue[loop2, loop1]
                # 求修正值
                
                C = error / sum(np.power(w, 2)) * w.conj()
                # 進行修正
                F = F +  C
        F[np.where(F < 0)] = 0
        c = c + 1

    F = F.reshape(size, size).conj()

    end = time.perf_counter()
    print('叠代耗時:%s 秒。'%(end - start))

    return F


if __name__ == '__main__':
    image = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)
    """
    theta: 射束旋轉角度
    num: 射束條數
    size: 圖片尺寸
    delta: 網格邊長
    ite_num: 叠代次數
    lam:松弛因子 
    """
    theta = np.linspace(0, 180, 45, dtype=np.float64)
    num = np.int64(256)
    size = np.int64(256)
    delta = np.int64(1)
    ite_num = np.int64(1)
    lam = np.float64(.25)
    sizeSquare = size * size

    # 計算投影值
    projectionValue = projection(image, theta)

    # pass filter
    projectionValueFiltered = ramFilter(theta, size)

    # 計算系統矩陣
    gridNum, gridLen = systeMartrix(theta, size, num, delta)

    # 重建後圖像矩陣
    F = np.zeros((size * size,))

    # 叠代法重建
    reconImage = iteration(theta, size, gridNum, gridLen, F, ite_num)
    print('共叠代%d次。'%ite_num)


    # 繪制原始圖像、重建圖像、誤差圖像
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(reconImage, cmap='gray')
    plt.title('Reconstruction Image')
    cv2.normalize(reconImage, reconImage, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("Reconstruction Image.png", reconImage)

    plt.subplot(1, 3, 3)
    plt.imshow(reconImage - image, cmap='gray')
    plt.title('Error Image')

    plt.savefig("ART.png", cmap='gray')

    plt.show()