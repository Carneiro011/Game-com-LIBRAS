import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- CONFIGURAÇÕES INICIAIS ---

# 1. INICIALIZAR O MEDIAPIPE
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,       # Processa um fluxo de vídeo
    max_num_hands=1,               # Detecta apenas uma mão
    min_detection_confidence=0.7,  # Confiança mínima para detecção
    min_tracking_confidence=0.5    # Confiança mínima para rastreamento
)

# 2. CARREGAR O MODELO TREINADO
# CERTIFIQUE-SE QUE O ARQUIVO 'keras_model.h5' ESTÁ NA MESMA PASTA
try:
    model = load_model('keras_model.h5')
    # Lista de letras que o modelo consegue reconhecer (ordem é importante!)
    # Esta ordem corresponde às saídas do modelo que baixamos.
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
except IOError:
    print("Erro: Arquivo 'keras_model.h5' não encontrado.")
    print("Por favor, baixe o modelo e coloque-o na mesma pasta do script.")
    exit()

# 3. CONFIGURAÇÕES DO JOGO
palavras = ["BOLA", "GATO", "SOL", "VIDA"]
palavra_atual_index = 0
letra_atual_index = 0
pontos = 0
feedback_visual = ""
feedback_timer = 0

# --- FUNÇÕES AUXILIARES ---

def preparar_imagem_para_modelo(hand_img):
    """Redimensiona e formata a imagem da mão para o modelo."""
    # O modelo foi treinado com imagens 224x224
    img_resized = cv2.resize(hand_img, (224, 224))
    # Normaliza a imagem (valores de pixel entre 0 e 1)
    img_normalized = img_resized / 255.0
    # Adiciona uma dimensão extra para o batch (formato esperado pelo modelo)
    img_reshaped = np.reshape(img_normalized, (1, 224, 224, 3))
    return img_reshaped

def desenhar_interface(frame, palavra, letra_idx, pred_letra, pontos, feedback):
    """Desenha toda a informação do jogo na tela."""
    height, width, _ = frame.shape
    
    # Desenha a palavra a ser soletrada
    palavra_display = ""
    for i, letra in enumerate(palavra):
        if i < letra_idx:
            # Letras já acertadas em verde
            palavra_display += f" {letra}"
        elif i == letra_idx:
            # Letra atual em destaque
            palavra_display += f" [{letra}]"
        else:
            # Próximas letras
            palavra_display += f" {letra}"

    cv2.putText(frame, "SOLETRE: " + palavra_display, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    
    # Desenha a pontuação
    cv2.putText(frame, f"PONTOS: {pontos}", (width - 250, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    
    # Desenha a letra prevista pelo modelo
    cv2.putText(frame, f"Previsto: {pred_letra}", (50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
    
    # Mostra feedback ("ACERTOU!")
    if feedback:
        cv2.putText(frame, feedback, (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)


# --- LOOP PRINCIPAL DO JOGO ---

cap = cv2.VideoCapture(0) # Inicia a câmera (0 é a câmera padrão)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inverte o frame horizontalmente (efeito espelho)
    frame = cv2.flip(frame, 1)
    
    # Converte a cor do frame de BGR para RGB (necessário para o MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processa o frame com o MediaPipe
    results = hands.process(frame_rgb)
    
    letra_prevista = ""
    
    # Se uma mão for detectada
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha os pontos e conexões da mão
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lógica para recortar a imagem da mão
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords) * w) - 20, int(max(x_coords) * w) + 20
            y_min, y_max = int(min(y_coords) * h) - 20, int(max(y_coords) * h) + 20
            
            # Garante que as coordenadas não saiam da tela
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            # Recorta a imagem da mão
            hand_image = frame[y_min:y_max, x_min:x_max]

            if hand_image.size > 0:
                # Prepara a imagem e faz a previsão
                img_para_previsao = preparar_imagem_para_modelo(hand_image)
                prediction = model.predict(img_para_previsao)
                
                # Obtém a letra com maior probabilidade
                predicted_index = np.argmax(prediction)
                letra_prevista = labels[predicted_index]

                # LÓGICA DO JOGO: VERIFICA SE O SINAL ESTÁ CORRETO
                palavra_alvo = palavras[palavra_atual_index]
                letra_alvo = palavra_alvo[letra_atual_index]

                # O modelo não reconhece todas as letras (ex: 'O' de BOLA). Pulamos se não estiver na lista.
                # Esta é uma limitação do modelo que baixamos, mas a lógica do jogo está aqui.
                if letra_alvo not in labels:
                    letra_atual_index += 1 # Pula para a próxima letra
                    if letra_atual_index >= len(palavra_alvo):
                        # Troca de palavra
                        palavra_atual_index = (palavra_atual_index + 1) % len(palavras)
                        letra_atual_index = 0
                elif letra_prevista == letra_alvo:
                    pontos += 10
                    letra_atual_index += 1
                    feedback_visual = "ACERTOU!"
                    feedback_timer = 30 # Mostra o feedback por 30 frames

                    # Se completou a palavra
                    if letra_atual_index >= len(palavra_alvo):
                        pontos += 50 # Bônus por completar a palavra
                        palavra_atual_index = (palavra_atual_index + 1) % len(palavras)
                        letra_atual_index = 0
                        feedback_visual = "PALAVRA COMPLETA!"
                        feedback_timer = 60
            
            # Desenha um retângulo ao redor da mão detectada
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Desenha a interface do jogo na tela
    if feedback_timer > 0:
        desenhar_interface(frame, palavras[palavra_atual_index], letra_atual_index, letra_prevista, pontos, feedback_visual)
        feedback_timer -= 1
    else:
        desenhar_interface(frame, palavras[palavra_atual_index], letra_atual_index, letra_prevista, pontos, "")


    # Mostra o resultado
    cv2.imshow('Herói do Alfabeto - Jogo em Libras', frame)

    # Verifica se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- FINALIZAÇÃO ---
hands.close()
cap.release()
cv2.destroyAllWindows()
