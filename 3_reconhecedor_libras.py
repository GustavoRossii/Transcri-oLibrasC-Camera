import cv2
import mediapipe as mp
import pickle
import numpy as np

try:
    with open('modelo_libras.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Erro: Arquivo 'modelo_libras.pkl' não encontrado.")
    print("Por favor, execute o script '2_treinador_modelo.py' primeiro.")
    exit()

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

frase = ""
letra_anterior = ""
contador_estabilidade = 0
LIMIAR_ESTABILIDADE = 15

print("Pressione 'ESPAÇO' para adicionar um espaço.")
print("Pressione 'BACKSPACE' (tecla de apagar) para corrigir.")
print("Pressione 'ESC' para limpar a frase.")
print("Pressione 'Q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    letra_atual = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            pontos_normalizados = []
            pulso_x = hand_landmarks.landmark[0].x
            pulso_y = hand_landmarks.landmark[0].y
            for landmark in hand_landmarks.landmark:
                pontos_normalizados.append(landmark.x - pulso_x)
                pontos_normalizados.append(landmark.y - pulso_y)


            prediction = model.predict([pontos_normalizados])
            confidence = model.predict_proba([pontos_normalizados])

            letra_prevista = prediction[0]
            confianca_previsao = np.max(confidence)


            if confianca_previsao > 0.5:
                letra_atual = letra_prevista

                if letra_atual == letra_anterior:
                    contador_estabilidade += 1
                else:
                    contador_estabilidade = 0
                    letra_anterior = letra_atual

                if contador_estabilidade == LIMIAR_ESTABILIDADE:
                    if len(frase) == 0 or (len(frase) > 0 and frase[-1] != letra_atual):
                        frase += letra_atual
                        contador_estabilidade = 0

            coords = hand_landmarks.landmark[0]
            x = int(coords.x * frame.shape[1]) - 50
            y = int(coords.y * frame.shape[0]) - 50
            cv2.putText(frame_copy, f'{letra_atual} ({confianca_previsao * 100:.2f}%)', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        contador_estabilidade = 0
        letra_anterior = ""

    h, w, _ = frame_copy.shape
    cv2.rectangle(frame_copy, (0, h - 60), (w, h), (255, 255, 255), -1)

    cv2.putText(frame_copy, frase, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Reconhecedor de Libras com Transcrição', frame_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        frase += " "
    elif key == 8:
        frase = frase[:-1]
    elif key == 27:
        frase = ""

cap.release()
cv2.destroyAllWindows()