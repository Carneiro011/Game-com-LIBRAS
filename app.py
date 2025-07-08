import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# --- CONFIGURAÇÕES INICIAIS ---

# Inicializa o MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Verificação e carregamento do modelo e labels
modelo_path = 'model_epoch_48_98.6_final.h5'
labels_path = 'labels.txt'

if not os.path.exists(modelo_path):
    print(f"❌ Modelo '{modelo_path}' não encontrado.")
    exit()

if not os.path.exists(labels_path):
    print(f"❌ Arquivo '{labels_path}' não encontrado.")
    exit()

try:
    model = load_model(modelo_path)
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"✅ Modelo carregado com sucesso! Labels: {len(labels)} classes.")
except Exception as e:
    print(f"Erro ao carregar modelo ou labels: {e}")
    exit()

# Palavras do jogo
palavras = ["BOLA", "GATO", "SOL", "VIDA"]
palavra_atual_index = 0
letra_atual_index = 0
pontos = 0
feedback_visual = ""
feedback_timer = 0

# --- FUNÇÕES ---

def preparar_imagem_para_modelo(hand_img):
    """Prepara imagem da mão para o modelo (64x64, grayscale)."""
    img_resized = cv2.resize(hand_img, (64, 64))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_normalized = img_gray / 255.0
    img_reshaped = np.reshape(img_normalized, (1, 64, 64, 1))
    return img_reshaped

def desenhar_interface(frame, palavra, letra_idx, pred_letra, pontos, feedback):
    height, width, _ = frame.shape

    palavra_display = ""
    for i, letra in enumerate(palavra):
        if i < letra_idx:
            palavra_display += f" {letra}"
        elif i == letra_idx:
            palavra_display += f" [{letra}]"
        else:
            palavra_display += f" {letra}"

    cv2.putText(frame, "SOLETRE: " + palavra_display, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"PONTOS: {pontos}", (width - 250, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Previsto: {pred_letra}", (50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    if feedback:
        cv2.putText(frame, feedback, (width // 2 - 150, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

# --- LOOP PRINCIPAL ---

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Não foi possível acessar a câmera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame não capturado.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    letra_prevista = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min, x_max = int(min(x_coords) * w) - 20, int(max(x_coords) * w) + 20
            y_min, y_max = int(min(y_coords) * h) - 20, int(max(y_coords) * h) + 20

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            hand_image = frame[y_min:y_max, x_min:x_max]

            if hand_image.size > 0:
                img_para_previsao = preparar_imagem_para_modelo(hand_image)
                prediction = model.predict(img_para_previsao)
                
                confidence = np.max(prediction)
                predicted_index = np.argmax(prediction)

                if confidence >= 0.7:
                    letra_prevista = labels[predicted_index]
                    print(f"🔤 Letra prevista: {letra_prevista} (confiança: {confidence:.2f})")
                else:
                    letra_prevista = ""

                palavra_alvo = palavras[palavra_atual_index]
                letra_alvo = palavra_alvo[letra_atual_index]

                if letra_alvo not in labels:
                    letra_atual_index += 1
                    if letra_atual_index >= len(palavra_alvo):
                        palavra_atual_index = (palavra_atual_index + 1) % len(palavras)
                        letra_atual_index = 0
                elif letra_prevista == letra_alvo:
                    pontos += 10
                    letra_atual_index += 1
                    feedback_visual = "ACERTOU!"
                    feedback_timer = 30

                    if letra_atual_index >= len(palavra_alvo):
                        pontos += 50
                        palavra_atual_index = (palavra_atual_index + 1) % len(palavras)
                        letra_atual_index = 0
                        feedback_visual = "PALAVRA COMPLETA!"
                        feedback_timer = 60

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    if feedback_timer > 0:
        desenhar_interface(frame, palavras[palavra_atual_index], letra_atual_index, letra_prevista, pontos, feedback_visual)
        feedback_timer -= 1
    else:
        desenhar_interface(frame, palavras[palavra_atual_index], letra_atual_index, letra_prevista, pontos, "")

    cv2.imshow('🧠 Herói do Alfabeto - Jogo em Libras', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- FINALIZAÇÃO ---
hands.close()
cap.release()
cv2.destroyAllWindows()
