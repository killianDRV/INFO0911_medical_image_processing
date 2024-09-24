import numpy as np
from PIL import Image
import warnings
import cv2

def add_gaussian_noise(image, noise_level=0.1):
    img_array = np.array(image)
    noise = np.random.normal(loc=0, scale=noise_level, size=img_array.shape)
    noisy_img = img_array + noise * 255
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img, mode='L')

def add_salt_and_pepper_noise(image, prob=0.05):
    img_array = np.array(image)
    salt_pepper_noise = np.random.rand(*img_array.shape)
    img_array[salt_pepper_noise < prob / 2] = 0
    img_array[salt_pepper_noise > 1 - prob / 2] = 255
    return Image.fromarray(img_array, mode='L')

def add_speckle_noise(image, noise_level=0.1):
    img_array = np.array(image).astype(float) / 255.0
    noise = np.random.normal(loc=0, scale=noise_level, size=img_array.shape)
    noisy_img = img_array + img_array * noise
    noisy_img = np.clip(noisy_img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img, mode='L')


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1, ploton=False):
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)
    
    # Initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # Initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for ii in range(niter):
        # Calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # Conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

        # Update matrices
        E = gE * deltaE
        S = gS * deltaS

        # Subtract a copy that has been shifted 'North/West' by one pixel
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # Update the image
        imgout += gamma * (NS + EW)

    return np.clip(imgout, 0, 255).astype(np.uint8)

import cv2

def coherence_filter_image(image, sigma=11, str_sigma=11, blend=0.5, iter_n=4):
    img = image.copy()
    h, w = img.shape[:2]

    for i in range(iter_n):
        # Vérifie si l'image a 3 canaux (couleur) ou 1 canal (niveaux de gris)
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img  # Si l'image est déjà en niveaux de gris
        
        # Calcul des valeurs propres (eigenvalues) et vecteurs propres (eigenvectors)
        eigen = cv2.cornerEigenValsAndVecs(gray, str_sigma, 3)
        eigen = eigen.reshape(h, w, 3, 2)
        x, y = eigen[:, :, 1, 0], eigen[:, :, 1, 1]

        # Calcul des dérivées secondes
        gxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=sigma)
        gxy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=sigma)
        gyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=sigma)
        
        # Calcul de la diffusion guidée par les valeurs propres
        gvv = x * x * gxx + 2 * x * y * gxy + y * y * gyy
        m = gvv < 0

        # Erosion et dilatation
        ero = cv2.erode(img, None)
        dil = cv2.dilate(img, None)
        img1 = ero
        img1[m] = dil[m]

        # Mélange de l'image originale et de l'image traitée
        img = np.uint8(img * (1.0 - blend) + img1 * blend)
    
    return img
