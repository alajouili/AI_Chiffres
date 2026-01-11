import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# 1. Chargement des données MNIST (chiffres manuscrits)
print("⏳ Chargement des données...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Préparation des données (Normalisation)
# On transforme les valeurs de 0-255 (gris) vers 0-1 pour l'IA
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# On transforme les labels (ex: le chiffre '5') en vecteur (0,0,0,0,0,1,0,0,0,0)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3. Création du Cerveau (Réseau de Neurones / CNN)
model = models.Sequential([
    # Couche 1 : Détection de traits simples
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Couche 2 : Détection de formes complexes
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Couche 3 : Aplatir et décider
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 neurones de sortie (pour 0 à 9)
])

# 4. Compilation et Entraînement
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("🚀 Démarrage de l'entraînement (cela peut prendre 1 à 2 minutes)...")
# On entraîne sur 5 cycles (epochs)
model.fit(x_train, y_train, epochs=5, batch_size=64)

# 5. Sauvegarde
model.save('mnist.h5')
print("✅ Modèle entraîné et sauvegardé sous le nom 'mnist.h5'")