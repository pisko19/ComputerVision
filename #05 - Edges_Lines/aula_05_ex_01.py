import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para aplicar thresholding e exibir o resultado em um único plot
def apply_thresholds(image_path):
    # Carregar a imagem em escala de cinza
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Verificar se a imagem foi carregada corretamente
    if image is None:
        print(f"Erro ao carregar a imagem {image_path}")
        return

    # Valor de threshold
    thresh_value = 127
    max_value = 255

    # Aplicar os diferentes tipos de thresholding
    _, thresh_binary = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_BINARY)
    _, thresh_binary_inv = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_BINARY_INV)
    _, thresh_trunc = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_TRUNC)
    _, thresh_tozero = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_TOZERO)
    _, thresh_tozero_inv = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_TOZERO_INV)

    # Configurar o layout para exibir as imagens
    plt.figure(figsize=(10, 8))  # Tamanho da figura

    # Exibir a imagem original
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Exibir os resultados das operações de thresholding
    plt.subplot(2, 3, 2)
    plt.imshow(thresh_binary, cmap='gray')
    plt.title("THRESH_BINARY")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(thresh_binary_inv, cmap='gray')
    plt.title("THRESH_BINARY_INV")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(thresh_trunc, cmap='gray')
    plt.title("THRESH_TRUNC")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(thresh_tozero, cmap='gray')
    plt.title("THRESH_TOZERO")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(thresh_tozero_inv, cmap='gray')
    plt.title("THRESH_TOZERO_INV")
    plt.axis('off')

    # Ajustar o layout para evitar sobreposição
    plt.tight_layout()
    
    # Exibir o plot
    plt.show()

# Caminho para a imagem de entrada
image_path = '../images/mon1.bmp'  # Substitua 'mon1.bmp' pelo caminho da sua imagem

# Aplicar as operações de thresholding
apply_thresholds(image_path)