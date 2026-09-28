import cv2
import numpy as np

def dust_and_scratches(image_path, radius, threshold):
    img = cv2.imread(image_path)
    if img is None:
        return
    # 将 Photoshop 的半径转换为 OpenCV 滤波器的核大小 (Kernel Size)
    # PS 中的半径是指中心像素到边缘的距离，所以核大小为 (radius * 2 + 1)
    kernel_size = radius * 2 + 1
    # 全图应用中值滤波（计算出每个像素周围的“标准”中值）
    median = cv2.medianBlur(img, kernel_size)
    # 计算原图与中值图之间的绝对差值
    # 注意：需要转换为 int16 以防止无符号变体减法溢出
    diff = cv2.absdiff(img, median).astype(np.int16)
    
    # 根据阈值生成掩膜 (Mask)
    # 在三维颜色通道中，只要任意一个通道的差值大于阈值，就进行替换
    mask = np.any(diff > threshold, axis=2)
    
    # 差值大于阈值的像素用中值代替，其余保留原图像素
    result = img.copy()
    result[mask] = median[mask]
    
    return result

if __name__ == "__main__":
    img = dust_and_scratches("blurred_image.tif", 1, 0)
    cv2.imwrite('noised_image.tif', img)