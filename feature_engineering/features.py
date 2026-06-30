import cv2
import numpy as np
from scipy.stats import entropy
from skimage.feature import local_binary_pattern


def blur_score(gray):

    return cv2.Laplacian(gray, cv2.CV_64F).var()


def edge_density(gray):

    edges = cv2.Canny(gray, 100, 200)

    return np.sum(edges > 0) / edges.size


def brightness(gray):

    return np.mean(gray)


def image_entropy(gray):

    hist = cv2.calcHist([gray], [0], None, [256], [0,256])

    hist = hist.ravel()

    hist = hist / hist.sum()

    return entropy(hist)


def rgb_mean(img):

    return np.mean(img, axis=(0,1))


def rgb_std(img):

    return np.std(img, axis=(0,1))


def sobel_x_mean(gray):

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    return np.mean(np.abs(sobelx))


def sobel_y_mean(gray):

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    return np.mean(np.abs(sobely))


def fft_high_frequency(gray):

   
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.log1p(np.abs(fshift))

    h, w = gray.shape
    cx, cy = h // 2, w // 2


    radius = 30

    mask = np.ones((h, w), dtype=np.uint8)

    cv2.circle(mask, (cy, cx), radius, 0, -1)

    high_freq = magnitude * mask

    return np.mean(high_freq)

def fft_low_frequency(gray):

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.log1p(np.abs(fshift))

    h, w = gray.shape
    cx, cy = h // 2, w // 2

    radius = 30

    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.circle(mask, (cy, cx), radius, 1, -1)

    low_freq = magnitude * mask

    return np.mean(low_freq)

def lbp_mean(gray):

    radius = 1
    points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method='uniform'
    )

    return np.mean(lbp)

def lbp_variance(gray):

    radius = 1
    points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method='uniform'
    )

    return np.var(lbp)

def rms_contrast(gray):

    return np.std(gray)


def saturation_mean(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    return np.mean(hsv[:, :, 1])

def tenengrad(gray):

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    return np.mean(np.sqrt(gx**2 + gy**2))


def noise_estimate(gray):

    blur = cv2.GaussianBlur(gray, (3,3), 0)

    noise = gray.astype(np.float32) - blur.astype(np.float32)

    return np.std(noise)

def pixel_grid_score(gray):

    # Convert to float
    gray = np.float32(gray)

    # Fourier Transform
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shift)

    h, w = magnitude.shape

    center = magnitude[h//2-20:h//2+20,
                       w//2-20:w//2+20]

    total_energy = np.sum(magnitude)

    center_energy = np.sum(center)

    pixel_score = (total_energy - center_energy) / total_energy

    return pixel_score

def extract_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    features = []

    
    features.append(blur_score(gray))
    
    features.append(edge_density(gray))
   
    features.append(brightness(gray))
   
    features.append(image_entropy(gray))
   
    features.extend(rgb_mean(img))
    
    features.extend(rgb_std(img))
    
    features.append(sobel_x_mean(gray))
    features.append(sobel_y_mean(gray))
    
    features.append(fft_high_frequency(gray))
    features.append(fft_low_frequency(gray))
    features.append(lbp_mean(gray))
    features.append(lbp_variance(gray))
    
    features.append(rms_contrast(gray))
   
    features.append(saturation_mean(img))
    
    features.append(tenengrad(gray))
   
    features.append(noise_estimate(gray))
    features.append(pixel_grid_score(gray))

    
    return np.array(features)

