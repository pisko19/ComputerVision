import sys
import numpy as np
import cv2
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

def apply_averaging_filters(image):
    # Aplicar filtro de média 3x3
    imageAFilter3x3 = cv2.blur(image, (3, 3))

    # Aplicar filtro de média 5x5
    imageAFilter5x5 = cv2.blur(image, (5, 5))

    # Aplicar filtro de média 7x7
    imageAFilter7x7 = cv2.blur(image, (7, 7))

    # Aplicar filtro 5x5 três vezes
    for _ in range(3):
        imageAFilter5x5 = cv2.blur(imageAFilter5x5, (5, 5))

    # Aplicar filtro 7x7 três vezes
    imageAFilter7x7_3x = imageAFilter7x7.copy()
    for _ in range(3):
        imageAFilter7x7_3x = cv2.blur(imageAFilter7x7_3x, (7, 7))

    return imageAFilter3x3, imageAFilter5x5, imageAFilter7x7, imageAFilter7x7_3x

# Carregar a imagem
image = cv2.imread("../images/lena.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image file could not be opened!")
    exit(-1)

printImageFeatures(image)

# Aplicar os filtros
imageAFilter3x3, imageAFilter5x5, imageAFilter7x7, imageAFilter7x7_3x = apply_averaging_filters(image)

# Mostrar resultados
plt.figure(figsize=(12, 8))

# Imagem original
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Filtro 3x3
plt.subplot(2, 3, 2)
plt.imshow(imageAFilter3x3, cmap='gray')
plt.title("Average Filter 3x3")
plt.axis('off')

# Filtro 5x5
plt.subplot(2, 3, 3)
plt.imshow(imageAFilter5x5, cmap='gray')
plt.title("Average Filter 5x5 (3x)")
plt.axis('off')

# Filtro 7x7
plt.subplot(2, 3, 4)
plt.imshow(imageAFilter7x7, cmap='gray')
plt.title("Average Filter 7x7")
plt.axis('off')

# Filtro 7x7 aplicado 3 vezes
plt.subplot(2, 3, 5)
plt.imshow(imageAFilter7x7_3x, cmap='gray')
plt.title("Average Filter 7x7 (3x)")
plt.axis('off')

plt.tight_layout()
plt.show()
