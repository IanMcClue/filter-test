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
    lut_r = np.array([i for i in range(256)], dtype=np.uint8)
    lut_g = np.array([min(255, i * 0.95 + 12) for i in range(256)], dtype=np.uint8)
    lut_b = np.array([min(255, i * 0.85 + 20) for i in range(256)], dtype=np.uint8)
    return apply_lut(image, lut_r, lut_g, lut_b)

def apply_fujifilm_velvia(image):
    lut_r = np.array([min(255, i * 0.9 + 10) for i in range(256)], dtype=np.uint8)
    lut_g = np.array([min(255, i * 0.85 + 20) for i in range(256)], dtype=np.uint8)
    lut_b = np.array([i for i in range(256)], dtype=np.uint8)
    return apply_lut(image, lut_r, lut_g, lut_b)

def create_warm_lut():
    lut_r = np.array([min(255, i + 20) for i in range(256)], dtype=np.uint8)
    lut_g = np.array([i for i in range(256)], dtype=np.uint8)
    lut_b = np.array([max(0, i - 20) for i in range(256)], dtype=np.uint8)
    return lut_r, lut_g, lut_b

def apply_lut(image, lut_r, lut_g, lut_b):
    r, g, b = cv2.split(image)
    r = cv2.LUT(r, lut_r)
    g = cv2.LUT(g, lut_g)
    b = cv2.LUT(b, lut_b)
    return cv2.merge((r, g, b))

def apply_custom_filter(image):
    lut_r, lut_g, lut_b = create_warm_lut()
    return apply_lut(image, lut_r, lut_g, lut_b)
