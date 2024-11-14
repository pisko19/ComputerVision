import cv2
import numpy as np
import matplotlib.pyplot as plt

def printImageFeatures(image):
    # Imprimir características da imagem
    if len(image.shape) == 2:
        height, width = image.shape
        nchannels = 1
    else:
        height, width, nchannels = image.shape

    # imprimir algumas características
    print("Image Height: %d" % height)
    print("Image Width: %d" % width)
    print("Image Channels: %d" % nchannels)
    print("Number of elements: %d" % image.size)

def apply_filters(image):
    # Aplicar filtro de média 5x5
    avg_filter_5x5 = cv2.blur(image, (5, 5))

    # Aplicar filtro mediano 5x5
    median_filter_5x5 = cv2.medianBlur(image, 5)

    # Aplicar filtro Gaussiano 3x3
    gaussian_filter_3x3 = cv2.GaussianBlur(image, (3, 3), sigmaX=0)

    # Aplicar filtro Gaussiano 5x5
    gaussian_filter_5x5 = cv2.GaussianBlur(image, (5, 5), sigmaX=0)

    # Aplicar filtro Gaussiano 7x7
    gaussian_filter_7x7 = cv2.GaussianBlur(image, (7, 7), sigmaX=0)

    return avg_filter_5x5, median_filter_5x5, gaussian_filter_3x3, gaussian_filter_5x5, gaussian_filter_7x7

# Testar com as imagens
images_to_test = ['../images/Lena_Ruido.png', '../images/DETI_Ruido.png', '../images/fce5noi3.bmp', '../images/fce5noi4.bmp', '../images/fce5noi6.bmp', '../images/sta2.bmp', '../images/sta2noi1.bmp']

for image_path in images_to_test:
    print(f"Aplicando filtros na imagem: {image_path}")

    # Carregar a imagem
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Image file could not be opened!")
        continue

    printImageFeatures(image)

    # Aplicar filtros
    avg_filter_5x5, median_filter_5x5, gaussian_filter_3x3, gaussian_filter_5x5, gaussian_filter_7x7 = apply_filters(image)

    # Mostrar resultados
    plt.figure(figsize=(12, 8))

    # Imagem original
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Filtro de média 5x5
    plt.subplot(2, 3, 2)
    plt.imshow(avg_filter_5x5, cmap='gray')
    plt.title("Average Filter 5x5")
    plt.axis('off')

    # Filtro mediano 5x5
    plt.subplot(2, 3, 3)
    plt.imshow(median_filter_5x5, cmap='gray')
    plt.title("Median Filter 5x5")
    plt.axis('off')

    # Filtro Gaussiano 3x3
    plt.subplot(2, 3, 4)
    plt.imshow(gaussian_filter_3x3, cmap='gray')
    plt.title("Gaussian Filter 3x3")
    plt.axis('off')

    # Filtro Gaussiano 5x5
    plt.subplot(2, 3, 5)
    plt.imshow(gaussian_filter_5x5, cmap='gray')
    plt.title("Gaussian Filter 5x5")
    plt.axis('off')

    # Filtro Gaussiano 7x7
    plt.subplot(2, 3, 6)
    plt.imshow(gaussian_filter_7x7, cmap='gray')
    plt.title("Gaussian Filter 7x7")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
