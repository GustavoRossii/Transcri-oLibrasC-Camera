import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

csv_path = './data_libras/dados_maos.csv'
data = pd.read_csv(csv_path, header=None)

y = data[0]
X = data.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = SVC(kernel='rbf', probability=True)

print("Iniciando o treinamento do modelo...")
model.fit(X_train, y_train)
print("Treinamento finalizado!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisão do modelo: {accuracy * 100:.2f}%')

with open('modelo_libras.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo salvo com sucesso como 'modelo_libras.pkl'")