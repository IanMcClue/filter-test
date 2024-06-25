import cv2
import numpy as np

def add_grain(image, intensity=0.2):
    noise = np.random.normal(0, 255 * intensity, image.shape).astype(np.uint8)
    noisy_image = cv2.addWeighted(image, 1 - intensity, noise, intensity, 0)
    return noisy_image

def add_light_leak(image, leak_intensity=0.5, num_leaks=3):
    h, w = image.shape[:2]
    
    light_leak_image = image.astype(np.float32) / 255.0  # Normalize the image for blending
    
    for _ in range(num_leaks):
        # Random position and radius for the light leak
        leak_center_x = np.random.randint(0, w)
        leak_center_y = np.random.randint(0, h)
        leak_radius = np.random.randint(int(min(h, w) * 0.2), int(min(h, w) * 0.5))
        
        # Create a circular gradient for the light leak
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        gradient = np.exp(-((xx * (w / (2 * leak_radius)) + leak_center_x / w - 0.5) ** 2 + 
                            (yy * (h / (2 * leak_radius)) + leak_center_y / h - 0.5) ** 2))
        
        # Randomize the color of the light leak (red to orange)
        leak_color = np.array([1.0, np.random.uniform(0.2, 0.6), 0.0])  # Red to orange gradient
        light_leak = leak_color * gradient[..., np.newaxis]
        
        # Blend the light leak with the image
        light_leak_image = cv2.addWeighted(light_leak_image, 1, light_leak, leak_intensity, 0)
    
    light_leak_image = np.clip(light_leak_image, 0, 1)  # Ensure the image values are within the range [0, 1]
    light_leak_image = (light_leak_image * 255).astype(np.uint8)  # Convert back to uint8
    
    return light_leak_image

def add_vignette(image, strength=0.5):
    h, w = image.shape[:2]
    x, y = np.ogrid[:h, :w]

    # Create a distance map
    dist_map = np.sqrt((x - h / 2) ** 2 + (y - w / 2) ** 2)

    # Normalize and apply strength
    max_dist = np.max(dist_map)
    vignette = 1 - (dist_map / max_dist) * strength

    # Apply vignette effect to each channel
    vignette_image = np.dstack([image[:, :, i] * vignette for i in range(image.shape[2])])
    vignette_image = np.clip(vignette_image, 0, 255).astype(np.uint8)
    
    return vignette_image

def apply_vintage_effect(image):
    # Apply sepia tone
    sepia_image = apply_sepia(image)

    # Add a slight blur
    blurred_image = apply_blur(sepia_image, ksize=(5, 5))

    # Add grain
    grain_image = add_grain(blurred_image, intensity=0.2)

    # Add light leaks
    light_leak_image = add_light_leak(grain_image, leak_intensity=0.3, num_leaks=3)

    # Add vignette effect
    vintage_image = add_vignette(light_leak_image, strength=0.5)

    return vintage_image

# Example filter functions to add
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
