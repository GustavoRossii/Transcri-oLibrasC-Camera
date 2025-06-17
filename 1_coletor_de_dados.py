import cv2
import mediapipe as mp
import os
import csv
import time

LETRAS_PARA_TREINAR = ['E', 'S', 'C', 'O', 'B', 'A', 'R']
NUM_AMOSTRAS = 500
TEMPO_CAPTURA = 5

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

DATA_DIR = './data_libras'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

CSV_FILE_PATH = os.path.join(DATA_DIR, 'dados_maos.csv')

csv_file = open(CSV_FILE_PATH, 'w', newline='')
csv_writer = csv.writer(csv_file)

cap = cv2.VideoCapture(0)

for letra in LETRAS_PARA_TREINAR:
    print(f'Coletando dados para a letra: {letra}')

    for i in range(TEMPO_CAPTURA, 0, -1):
        ret, frame = cap.read()
        if not ret: continue

        cv2.putText(frame, f'Posicione a mao para a letra {letra}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)
        cv2.putText(frame, f'Capturando em... {i}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Coletor de Dados', cv2.flip(frame, 1))
        cv2.waitKey(1000)

    amostras_coletadas = 0
    while amostras_coletadas < NUM_AMOSTRAS:
        ret, frame = cap.read()
        if not ret: continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                pontos_normalizados = []
                pulso_x = hand_landmarks.landmark[0].x
                pulso_y = hand_landmarks.landmark[0].y

                for landmark in hand_landmarks.landmark:
                    pontos_normalizados.append(landmark.x - pulso_x)
                    pontos_normalizados.append(landmark.y - pulso_y)

                linha = [letra] + pontos_normalizados
                csv_writer.writerow(linha)
                amostras_coletadas += 1

        cv2.putText(frame, f'Coletando amostra {amostras_coletadas}/{NUM_AMOSTRAS} para a letra {letra}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Coletor de Dados', cv2.flip(frame, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Coleta de dados finalizada!")
cap.release()
csv_file.close()
cv2.destroyAllWindows()
