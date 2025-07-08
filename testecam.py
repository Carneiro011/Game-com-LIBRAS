import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Erro ao acessar a câmera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame inválido")
        break

    cv2.imshow("Teste de Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
