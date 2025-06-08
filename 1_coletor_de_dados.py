import cv2
import mediapipe as mp
import os
import csv
import time

# --- CONFIGURAÇÕES ---
# Defina as letras que você quer treinar. Adicione as letras do seu nome aqui.
LETRAS_PARA_TREINAR = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
NUM_AMOSTRAS = 300  # Número de amostras a coletar por letra
TEMPO_CAPTURA = 5  # Segundos de preparação antes de começar a capturar

# --- INICIALIZAÇÃO ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Cria o diretório para salvar os dados se não existir
DATA_DIR = './data_libras'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Nome do arquivo CSV para salvar os dados
CSV_FILE_PATH = os.path.join(DATA_DIR, 'dados_maos.csv')

# Abre o arquivo CSV para escrita
csv_file = open(CSV_FILE_PATH, 'w', newline='')
csv_writer = csv.writer(csv_file)

# --- CAPTURA DE DADOS ---
cap = cv2.VideoCapture(0)

for letra in LETRAS_PARA_TREINAR:
    print(f'Coletando dados para a letra: {letra}')

    # Pausa para o usuário se preparar
    for i in range(TEMPO_CAPTURA, 0, -1):
        ret, frame = cap.read()
        if not ret: continue

        cv2.putText(frame, f'Posicione a mao para a letra {letra}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)
        cv2.putText(frame, f'Capturando em... {i}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Coletor de Dados', cv2.flip(frame, 1))
        cv2.waitKey(1000)  # Espera 1 segundo

    # Coleta as amostras
    amostras_coletadas = 0
    while amostras_coletadas < NUM_AMOSTRAS:
        ret, frame = cap.read()
        if not ret: continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extrai e normaliza os pontos
                pontos_normalizados = []
                # Pega as coordenadas do pulso (ponto 0) como referência
                pulso_x = hand_landmarks.landmark[0].x
                pulso_y = hand_landmarks.landmark[0].y

                for landmark in hand_landmarks.landmark:
                    # Subtrai as coordenadas do pulso para tornar a posição relativa
                    pontos_normalizados.append(landmark.x - pulso_x)
                    pontos_normalizados.append(landmark.y - pulso_y)

                # Salva os dados no CSV
                linha = [letra] + pontos_normalizados
                csv_writer.writerow(linha)
                amostras_coletadas += 1

        # Mostra o progresso na tela
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
