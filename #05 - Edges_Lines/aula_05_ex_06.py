import cv2
import numpy as np

def main():
    # Captura de vídeo da câmera
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Erro ao abrir a câmera.")
        return

    # Criar janelas uma vez
    cv2.namedWindow("Detecção de Linhas - Câmera")
    cv2.namedWindow("Bordas Detectadas")

    while True:
        # Ler o frame da câmera
        ret, frame = capture.read()
        if not ret:
            print("Erro ao capturar frame.")
            break

        # Converter o frame para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar o detector de bordas de Canny
        edges = cv2.Canny(gray, 50, 150)

        # Usar a Transformada de Hough para detectar linhas
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

        # Desenhar as linhas detectadas no frame original
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # linha verde

        # Exibir o frame com as linhas detectadas
        cv2.imshow("Detecção de Linhas - Câmera", frame)

        # Exibir a imagem com as bordas detectadas
        cv2.imshow("Bordas Detectadas", edges)

        # Sair do loop quando a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a captura e fechar as janelas
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
