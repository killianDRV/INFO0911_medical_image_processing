import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def coherence_filter_image(image, sigma=11, str_sigma=11, blend=0.5, iter_n=4):
    img = image.copy()
    h, w = img.shape[:2]

    for i in range(iter_n):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eigen = cv2.cornerEigenValsAndVecs(gray, str_sigma, 3)
        eigen = eigen.reshape(h, w, 3, 2)
        x, y = eigen[:, :, 1, 0], eigen[:, :, 1, 1]

        gxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=sigma)
        gxy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=sigma)
        gyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=sigma)
        gvv = x * x * gxx + 2 * x * y * gxy + y * y * gyy
        m = gvv < 0

        ero = cv2.erode(img, None)
        dil = cv2.dilate(img, None)
        img1 = ero
        img1[m] = dil[m]
        img = np.uint8(img * (1.0 - blend) + img1 * blend)
    
    return img

# Exemple d'utilisation :
if __name__ == '__main__':
    # Lire l'image
    image_path = "arbre.png"  # Remplacez par le chemin réel
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Erreur : impossible de charger l'image '{image_path}'.")
        sys.exit(1)

    # Appliquer le filtre de cohérence
    filtered_img = coherence_filter_image(img, sigma=11, str_sigma=11, blend=0.5, iter_n=4)

    # Afficher les résultats avec matplotlib
    plt.subplot(1, 2, 1)
    plt.title('Image Originale')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convertir BGR en RGB pour matplotlib

    plt.subplot(1, 2, 2)
    plt.title('Image Filtrée')
    plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))  # Convertir BGR en RGB pour matplotlib

    plt.show()
