# src/hand_detector.py

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import os

def load_labels(path):
    """Carrega as labels (uma por linha) de um TXT."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

class HandDetector:
    def __init__(self, config):
        # --- MediaPipe Hands ---
        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.hands    = self.mp_hands.Hands(
            static_image_mode        = config.get("static_image_mode", False),
            max_num_hands            = config.get("max_num_hands", 1),
            min_detection_confidence = config.get("min_detection_confidence", 0.7),
            min_tracking_confidence  = config.get("min_tracking_confidence", 0.5)
        )

        # --- Carrega modelo e labels ---
        base = os.path.dirname(__file__)
        model_path  = os.path.abspath(os.path.join(base, config["model_path"]))
        labels_path = os.path.abspath(os.path.join(base, config["labels_file"]))

        print(f">>> Carregando modelo de: {model_path}")
        self.model = load_model(model_path)
        in_shape = self.model.input_shape  # e.g. (None, 64, 64, 1) ou (None,224,224,3)
        print(f"→ Input shape esperado: {in_shape}, número de classes: {self.model.output_shape[-1]}")

        self.labels = load_labels(labels_path)
        print("→ Labels:", self.labels)

        # extrai height, width, channels
        _, self.h_in, self.w_in, self.c_in = in_shape

        # --- Parâmetros de predição e suavização ---
        self.thresh  = config.get("prediction_threshold", 0.6)
        self.margin  = config.get("margin_threshold",    0.2)
        self.history = deque(maxlen=config.get("smoothing_window", 5))

        # --- Prepara CLAHE para equalização de histograma ---
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


    def _prepare(self, img):
        """
        1) Aplica CLAHE no canal Y para equalizar histograma.
        2) Converte para grayscale se canal único, ou RGB.
        3) Redimensiona para (w_in, h_in), normaliza e reshape.
        """
        # 1) CLAHE no YUV
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = self.clahe.apply(img_yuv[:,:,0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # 2) Converte de acordo com canais do modelo
        if self.c_in == 1:
            gray    = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (self.w_in, self.h_in))
            norm    = resized.astype(np.float32) / 255.0
            return norm.reshape(1, self.h_in, self.w_in, 1)
        else:
            rgb     = cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.w_in, self.h_in))
            norm    = resized.astype(np.float32) / 255.0
            return norm.reshape(1, self.h_in, self.w_in, self.c_in)


    def detect(self, frame):
        """
        1) Faz detecção de mãos no frame original.
        2) Escolhe a mão de maior área, faz crop limpo.
        3) Chama _prepare + model.predict + margin+threshold.
        4) Suavização condicionada e debug visual dos top-3.
        5) Desenha bbox e landmarks só na cópia.
        """
        vis     = frame.copy()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res     = self.hands.process(img_rgb)
        letter  = ""

        if res.multi_hand_landmarks:
            best_area = 0
            best_bbox = None

            # encontra maior bbox entre as mãos detectadas
            h, w, _ = frame.shape
            for lm in res.multi_hand_landmarks:
                xs = [p.x for p in lm.landmark]
                ys = [p.y for p in lm.landmark]
                x1 = max(int(min(xs)*w) - 20, 0)
                x2 = min(int(max(xs)*w) + 20, w)
                y1 = max(int(min(ys)*h) - 20, 0)
                y2 = min(int(max(ys)*h) + 20, h)
                area = (x2 - x1) * (y2 - y1)
                if area > best_area and (x2 > x1 and y2 > y1):
                    best_area = area
                    best_bbox = (x1, y1, x2, y2)

            # se achou bbox válida, classifica
            if best_bbox:
                x1, y1, x2, y2 = best_bbox
                crop = frame[y1:y2, x1:x2]
                inp  = self._prepare(crop)
                pred = self.model.predict(inp)[0]

                # --- Debug visual: mostra top-3 no vis ---
                idxs = np.argsort(pred)[::-1][:3]
                debug_text = "  ".join(f"{self.labels[i]}:{pred[i]:.2f}" for i in idxs)
                cv2.putText(vis, debug_text, (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                # top1 vs top2 para margin+threshold
                top, nxt  = idxs[0], idxs[1]
                conf_top  = pred[top]
                conf_next = pred[nxt]

                if conf_top >= self.thresh and (conf_top - conf_next) >= self.margin:
                    letter_raw = self.labels[top]
                    # só acumula no histórico se válida
                    self.history.append(letter_raw)

                # votação condicionada
                votes = [l for l in self.history if l]
                if votes:
                    letter = max(set(votes), key=votes.count)

                # desenha bbox no vis
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)

            # desenha landmarks na cópia para exibição
            for lm in res.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(vis, lm, self.mp_hands.HAND_CONNECTIONS)

        return vis, letter
