import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# --- CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
# Carrega os dados do arquivo CSV
csv_path = './data_libras/dados_maos.csv'
data = pd.read_csv(csv_path, header=None)

# Separa os dados em características (X) e rótulos (y)
# A primeira coluna (0) é o rótulo (a letra)
y = data[0]
# O resto das colunas são as características (coordenadas dos pontos)
X = data.iloc[:, 1:]

# Divide os dados em conjuntos de treino e teste (80% para treino, 20% para teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# --- TREINAMENTO DO MODELO ---
# Inicializa o modelo. SVC (Support Vector Classifier) é uma ótima escolha.
# kernel='rbf' é bom para padrões complexos.
# probability=True permite obter a confiança da previsão.
model = SVC(kernel='rbf', probability=True)

print("Iniciando o treinamento do modelo...")
# Treina o modelo com os dados de treino
model.fit(X_train, y_train)
print("Treinamento finalizado!")

# --- AVALIAÇÃO DO MODELO ---
# Faz previsões com os dados de teste
y_pred = model.predict(X_test)

# Calcula e imprime a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisão do modelo: {accuracy * 100:.2f}%')

# --- SALVANDO O MODELO TREINADO ---
# Usa a biblioteca pickle para salvar o objeto do modelo em um arquivo
with open('modelo_libras.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo salvo com sucesso como 'modelo_libras.pkl'")