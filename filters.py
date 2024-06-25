import cv2
import numpy as np

# Precomputed LUTs
lut_kodak_portra = None
lut_fujifilm_velvia = None
lut_cinestill_800t = None
lut_kodak_ektar_100 = None
lut_fujifilm_provia_100f = None

def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, kernel)
    return sepia_image

def apply_blur(image, ksize=(15, 15)):
    return cv2.GaussianBlur(image, ksize, 0)

def apply_edge_detection(image):
    return cv2.Canny(image, 100, 200)

def apply_kodak_portra(image):
    global lut_kodak_portra
    if lut_kodak_portra is None:
        lut_r = np.array([i for i in range(256)], dtype=np.uint8)
        lut_g = np.array([min(255, i * 0.95 + 12) for i in range(256)], dtype=np.uint8)
        lut_b = np.array([min(255, i * 0.85 + 20) for i in range(256)], dtype=np.uint8)
        lut_kodak_portra = lut_r, lut_g, lut_b
    return apply_lut(image, *lut_kodak_portra)

def apply_fujifilm_velvia(image):
    global lut_fujifilm_velvia
    if lut_fujifilm_velvia is None:
        lut_r = np.array([min(255, i * 0.9 + 10) for i in range(256)], dtype=np.uint8)
        lut_g = np.array([min(255, i * 0.85 + 20) for i in range(256)], dtype=np.uint8)
        lut_b = np.array([i for i in range(256)], dtype=np.uint8)
        lut_fujifilm_velvia = lut_r, lut_g, lut_b
    return apply_lut(image, *lut_fujifilm_velvia)

def apply_cinestill_800t(image):
    global lut_cinestill_800t
    if lut_cinestill_800t is None:
        lut_r = np.array([min(255, i * 0.9 + 20) for i in range(256)], dtype=np.uint8)
        lut_g = np.array([min(255, i * 0.85 + 15) for i in range(256)], dtype=np.uint8)
        lut_b = np.array([max(0, min(255, i * 1.1 - 10)) for i in range(256)], dtype=np.uint8)
        lut_cinestill_800t = lut_r, lut_g, lut_b
    return apply_lut(image, *lut_cinestill_800t)

def apply_kodak_ektar_100(image):
    global lut_kodak_ektar_100
    if lut_kodak_ektar_100 is None:
        lut_r = np.array([min(255, i * 1.1) for i in range(256)], dtype=np.uint8)
        lut_g = np.array([min(255, i * 0.95) for i in range(256)], dtype=np.uint8)
        lut_b = np.array([max(0, min(255, i * 0.9 + 10)) for i in range(256)], dtype=np.uint8)
        lut_kodak_ektar_100 = lut_r, lut_g, lut_b
    return apply_lut(image, *lut_kodak_ektar_100)

def apply_fujifilm_provia_100f(image):
    global lut_fujifilm_provia_100f
    if lut_fujifilm_provia_100f is None:
        lut_r = np.array([min(255, i * 1.05) for i in range(256)], dtype=np.uint8)
        lut_g = np.array([min(255, i * 0.95 + 10) for i in range(256)], dtype=np.uint8)
        lut_b = np.array([max(0, min(255, i * 1.1 - 20)) for i in range(256)], dtype=np.uint8)
        lut_fujifilm_provia_100f = lut_r, lut_g, lut_b
    return apply_lut(image, *lut_fujifilm_provia_100f)

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

def apply_vintage_effect(image):
    # Apply sepia tone
    sepia_image = apply_sepia(image)

    # Add a slight blur
    blurred_image = apply_blur(sepia_image, ksize=(5, 5))

    # Add noise
    noise = np.random.normal(0, 10, blurred_image.shape)
    noisy_image = blurred_image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Add vignette effect (optional)
    vignette_image = add_vignette(noisy_image, 0.5, 0.5)

    return vignette_image

def add_vignette(image, amount=0.5, midpoint=0.5):
    h, w = image.shape[:2]
    x, y = np.ogrid[:h, :w]

    # Create a distance map
    dist_map = np.sqrt((x - h / 2) ** 2 + (y - w / 2) ** 2)

    # Calculate the vignette effect
    max_dist = np.max(dist_map)
    vignette = 1 - (1 - amount) * (dist_map / max_dist) ** midpoint

    # Apply the vignette effect to each channel
    vignette_image = np.dstack([image[:, :, i] * vignette for i in range(image.shape[2])])

    # Normalize the image to [0, 1] range
    vignette_image = vignette_image / 255.0  # Assuming image was clipped to [0, 255] before

    return vignette_image
