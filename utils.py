import numpy as np
from PIL import Image
import warnings
import cv2
import base64
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

def add_gaussian_noise(image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """
    Add Gaussian noise to an image.
    
    Args:
        image (numpy.ndarray): Input image.
        noise_level (float): Standard deviation of the Gaussian noise. Default is 0.1.
        
    Returns:
        numpy.ndarray: Image with added Gaussian noise.
    """
    img_array = np.array(image)
    noise = np.random.normal(loc=0, scale=noise_level, size=img_array.shape)
    noisy_img = img_array + noise * 255
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img, mode='L')

def add_salt_and_pepper_noise(image: np.ndarray, prob: float =0.05) -> np.ndarray:
    """
    Add salt and pepper noise to an image.
    
    Args:
        image (numpy.ndarray): Input image.
        prob (float): Probability of a pixel being affected by noise. Default is 0.05.
        
    Returns:
        numpy.ndarray: Image with added salt and pepper noise.
    """
    img_array = np.array(image)
    salt_pepper_noise = np.random.rand(*img_array.shape)
    img_array[salt_pepper_noise < prob / 2] = 0
    img_array[salt_pepper_noise > 1 - prob / 2] = 255
    return Image.fromarray(img_array, mode='L')

def add_speckle_noise(image: np.ndarray, noise_level: float =0.1) -> np.ndarray:
    """
    Add speckle noise to an image.
    
    Args:
        image (numpy.ndarray): Input image.
        noise_level (float): Standard deviation of the multiplicative noise. Default is 0.1.
        
    Returns:
        numpy.ndarray: Image with added speckle noise.
    """
    img_array = np.array(image).astype(float) / 255.0
    noise = np.random.normal(loc=0, scale=noise_level, size=img_array.shape)
    noisy_img = img_array + img_array * noise
    noisy_img = np.clip(noisy_img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img, mode='L')


def anisodiff(img: np.ndarray, niter: int =1, kappa: float=50, gamma: float=0.1, step: tuple=(1., 1.), option: int=1) -> np.ndarray:
    """
    Add Perona-Malik anisotropic diffusion filter.
    
    Args:
        img (numpy.ndarray): Input image
        niter (int): Number of iterations
        kappa (float): Edge sensitivity parameter
        gamma (float): Learning rate
        step (tuple): No diffusion
        option (int): Diffusion function type (1 or 2)
        
    Returns:
        numpy.ndarray: Filtered image
    """
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

def coherence_filter_image(image: np.ndarray, sigma: int=11, str_sigma: int=11, blend: float=0.5, iter_n: int=4) -> np.ndarray:
    """
    Apply a coherence-enhancing filter to an image.
    
    Args:
        image (numpy.ndarray): Input image.
        sigma (int): Standard deviation for Gaussian smoothing. Default is 11.
        str_sigma (int): Structure tensor integration scale. Default is 11.
        blend (float): Blending factor between original and filtered image. Default is 0.5.
        iter_n (int): Number of iterations. Default is 4.
        
    Returns:
        numpy.ndarray: Filtered image with enhanced coherent structures.
    """
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

def formate_image(image: np.ndarray) -> np.ndarray:
    """
    Crop margins from an image.
    
    Args:
        image (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Cropped image with 50-pixel margins removed from each side.
    """
    margin = 50
    height, width = image.shape[:2]

    cropped_image = image[margin:height-margin, margin:width-margin]

    return cropped_image

def find_contours(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Find and draw contours in an image based on adaptive thresholding.
    
    Args:
        image (numpy.ndarray): Input grayscale image.
        threshold (int): Threshold value for binary thresholding.
        
    Returns:
        numpy.ndarray: Binary image with filled contours below the specified threshold.
    """
    equalized_image = cv2.equalizeHist(image)

    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        pleural_line_image = np.zeros_like(image)
        cv2.drawContours(pleural_line_image, [largest_contour], -1, (255), thickness=cv2.FILLED)

        _, binary_below = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        contours_below, _ = cv2.findContours(binary_below, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output_below = np.zeros_like(image)

        cv2.drawContours(output_below, contours_below, -1, (255), thickness=cv2.FILLED)

        return output_below
    else:
        print("Aucun contour trouvé.")

def calculate_scores(original: np.ndarray, compared: np.ndarray) -> str:
    """
    Calculate image quality metrics between original and compared images.
    
    Args:
        original (numpy.ndarray): Original reference image.
        compared (numpy.ndarray): Image to compare against the original.
        
    Returns:
        str: HTML-formatted string containing PSNR, MSE, and SSIM scores.
    """
    psnr_score = psnr(original, compared)
    mse_score = mse(original, compared)
    ssim_score, _ = ssim(original, compared, full=True)
    return f"<br>PSNR: {psnr_score:.2f}<br>MSE: {mse_score:.2f}<br>SSIM: {ssim_score:.2f}"

def cv2_to_base64(image: np.ndarray) -> str:
    """
    Convert an OpenCV image to base64 string representation.
    
    Args:
        image (numpy.ndarray): OpenCV image to convert.
        
    Returns:
        str: Base64 encoded string of the image in PNG format.
    """
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')