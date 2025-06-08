import cv2
import mediapipe as mp
import pickle
import numpy as np

# --- CARREGAMENTO DO MODELO ---
try:
    with open('modelo_libras.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Erro: Arquivo 'modelo_libras.pkl' não encontrado.")
    print("Por favor, execute o script '2_treinador_modelo.py' primeiro.")
    exit()

# --- INICIALIZAÇÃO DA CÂMERA E MEDIAPIPE ---
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# --- NOVAS VARIÁVEIS PARA TRANSCRIÇÃO ---
frase = ""
letra_anterior = ""
contador_estabilidade = 0
LIMIAR_ESTABILIDADE = 15  # Número de frames que uma letra precisa ficar estável para ser adicionada

print("Pressione 'ESPAÇO' para adicionar um espaço.")
print("Pressione 'BACKSPACE' (tecla de apagar) para corrigir.")
print("Pressione 'ESC' para limpar a frase.")
print("Pressione 'Q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inverte a imagem para efeito de espelho
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()  # Cria uma cópia para desenhar sobre

    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    letra_atual = ""  # Reseta a letra atual em cada frame

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extrai e normaliza os pontos (mesma lógica de antes)
            pontos_normalizados = []
            pulso_x = hand_landmarks.landmark[0].x
            pulso_y = hand_landmarks.landmark[0].y
            for landmark in hand_landmarks.landmark:
                pontos_normalizados.append(landmark.x - pulso_x)
                pontos_normalizados.append(landmark.y - pulso_y)

            # Faz a previsão
            prediction = model.predict([pontos_normalizados])
            confidence = model.predict_proba([pontos_normalizados])

            letra_prevista = prediction[0]
            confianca_previsao = np.max(confidence)

            # Lógica de estabilidade
            if confianca_previsao > 0.4:  # Aumentamos um pouco a confiança necessária
                letra_atual = letra_prevista

                # Se a letra detectada é a mesma da anterior, aumenta o contador
                if letra_atual == letra_anterior:
                    contador_estabilidade += 1
                else:
                    # Se mudou a letra, reseta o contador e atualiza a letra anterior
                    contador_estabilidade = 0
                    letra_anterior = letra_atual

                # Se o contador atingir o limiar, adiciona a letra à frase
                if contador_estabilidade == LIMIAR_ESTABILIDADE:
                    # Adiciona a letra apenas se for diferente da última letra da frase
                    if len(frase) == 0 or (len(frase) > 0 and frase[-1] != letra_atual):
                        frase += letra_atual
                        contador_estabilidade = 0  # Zera para não adicionar de novo

            # Mostra a previsão atual na tela
            coords = hand_landmarks.landmark[0]
            x = int(coords.x * frame.shape[1]) - 50
            y = int(coords.y * frame.shape[0]) - 50
            cv2.putText(frame_copy, f'{letra_atual} ({confianca_previsao * 100:.2f}%)', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        # Se nenhuma mão for detectada, reseta a estabilidade
        contador_estabilidade = 0
        letra_anterior = ""

    # --- DESENHA A INTERFACE DE TRANSCRIÇÃO ---
    # Cria uma barra branca na parte de baixo da tela
    h, w, _ = frame_copy.shape
    cv2.rectangle(frame_copy, (0, h - 60), (w, h), (255, 255, 255), -1)

    # Escreve a frase na barra branca
    cv2.putText(frame_copy, frase, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Mostra a imagem resultante na tela
    cv2.imshow('Reconhecedor de Libras com Transcrição', frame_copy)

    # --- CONTROLES DO TECLADO ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Tecla de Espaço
        frase += " "
    elif key == 8:  # Tecla Backspace (Apagar)
        frase = frase[:-1]
    elif key == 27:  # Tecla Esc
        frase = ""

cap.release()
cv2.destroyAllWindows()