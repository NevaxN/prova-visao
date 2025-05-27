"""-- Parte 1 --"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

pasta_imagens = "./imagens"

mapa_classes = {
    "gato": 0,
    "cachorro": 1
}

def preprocessar_imagem(caminho):
    img_bgr = cv2.imread(caminho)
    if img_bgr is None:
        print(f"Erro ao carregar {caminho}")
        return None, None, None
    
    # Redimensionar para 128x128
    img_redim = cv2.resize(img_bgr, (128, 128))
    
    # Converter para cinza
    img_gray = cv2.cvtColor(img_redim, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro Gaussiano
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Equalização de histograma
    img_eq = cv2.equalizeHist(img_blur)
    
    return img_redim, img_blur, img_eq

imagens_originais = []
imagens_filtro = []
imagens_eq = []
labels = []

for nome_arquivo in os.listdir(pasta_imagens):
    nome_arquivo_lower = nome_arquivo.lower()
    if nome_arquivo_lower.endswith(('.jpg', '.png')):
        caminho = os.path.join(pasta_imagens, nome_arquivo)
        img_orig, img_blur, img_eq = preprocessar_imagem(caminho)
        if img_orig is not None:
            imagens_originais.append(img_orig)
            imagens_filtro.append(img_blur)
            imagens_eq.append(img_eq)
            
            if "gato" in nome_arquivo_lower:
                labels.append(mapa_classes["gato"])
            elif "cachorro" in nome_arquivo_lower:
                labels.append(mapa_classes["cachorro"])

# Mostrar as imagens: original, com filtro e equalizada (cinza)
for i in range(len(imagens_originais)):
    plt.figure(figsize=(10,3))
    
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(imagens_originais[i], cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow(imagens_filtro[i], cmap='gray')
    plt.title('Filtro Gaussiano')
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.imshow(imagens_eq[i], cmap='gray')
    plt.title('Equalização Histograma')
    plt.axis('off')
    
    plt.show()

"""-- Parte 2 --"""
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# --- Carregar CIFAR-10 gatos e cachorros ---
(X_full, y_full), _ = tf.keras.datasets.cifar10.load_data()

def filtrar_cats_dogs(X, y):
    mask = (y.flatten() == 3) | (y.flatten() == 5)
    X, y = X[mask], y[mask]
    y = (y == 5).astype(np.uint8)  # gato -> 0, cachorro -> 1
    return X, y

X, y = filtrar_cats_dogs(X_full, y_full)

# Normalização (sem redimensionar, CIFAR-10 já é 32x32)
X = X.astype('float32') / 255.0

# Dividir em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treinamento mais rápido
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Avaliação
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=["Gato", "Cachorro"]))

# --- Predição nas imagens externas (sem filtros) ---
print("\nPredição nas imagens externas (sem filtros):")

pasta_imagens = "./imagens"
class_names = ["Gato", "Cachorro"]

for nome_arquivo in os.listdir(pasta_imagens):
    if nome_arquivo.lower().endswith(('.jpg', '.png')):
        caminho = os.path.join(pasta_imagens, nome_arquivo)
        
        img_bgr = cv2.imread(caminho)
        if img_bgr is None:
            print(f"Erro ao carregar {nome_arquivo}")
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_redim = cv2.resize(img_rgb, (32, 32))
        img_input = img_redim.astype('float32') / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        prob = model.predict(img_input)[0][0]
        classe = "Cachorro" if prob > 0.5 else "Gato"
        confianca = prob if prob > 0.5 else 1 - prob
        
        print(f"{nome_arquivo}: {classe} (confiança: {confianca:.2%})")
        
        plt.imshow(img_rgb)
        plt.title(f"{nome_arquivo} → {classe}")
        plt.axis('off')
        plt.show()