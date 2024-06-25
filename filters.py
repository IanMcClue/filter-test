import cv2
import numpy as np

def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.transform(image, kernel)

def apply_blur(image, ksize=(15, 15)):
    return cv2.GaussianBlur(image, ksize, 0)

def apply_edge_detection(image):
    return cv2.Canny(image, 100, 200)

def apply_kodak_portra(image):
    lut = np.array([[i, i * 0.95 + 12, i * 0.85 + 20] for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)

def apply_fujifilm_velvia(image):
    lut = np.array([[i * 0.9 + 10, i * 0.85 + 20, i] for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)

def create_warm_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lut[i] = [min(255, i + 20), i, max(0, i - 20)]
    return lut

def apply_lut(image, lut):
    r, g, b = cv2.split(image)
    r = cv2.LUT(r, lut[:, 0])
    g = cv2.LUT(g, lut[:, 1])
    b = cv2.LUT(b, lut[:, 2])
    return cv2.merge((r, g, b))

def apply_custom_filter(image):
    lut = create_warm_lut()
    return apply_lut(image, lut)
