import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CHARGEMENT DES MÉTADONNÉES ET DES DONNÉES
# ==========================================

# Lecture du fichier de configuration de ton collègue
with open('data/processing/cifar-10/meta_animals.json', 'r') as f:
    meta_animals = json.load(f)

# Extraction des informations structurelles
classes_animaux = meta_animals['label_names']       # ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
input_shape = tuple(meta_animals['image_shape'])    # (32, 32, 3)
num_classes = len(classes_animaux)                 # 6

print(f"Classes cibles pour le projet : {classes_animaux}")
print(f"Format d'entrée des images : {input_shape}")

# Chargement des datasets filtrés (.npz)
train_archive = np.load('data/processing/cifar-10/train_animals.npz')
test_archive = np.load('data/processing/cifar-10/test_animals.npz')

# Extraction (en utilisant les clés standards NumPy par défaut 'X' et 'y')
X_train_full = train_archive['X']
y_train_full = train_archive['y']
X_test = test_archive['X']
y_test = test_archive['y']

# Séparation Train / Validation (20% pour la validation, stratifié sur y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"\n[Data Split] Train: {X_train.shape} | Validation: {X_val.shape} | Test: {X_test.shape}")


# ==========================================
# 2. MODÈLE 1 : MULTILAYER PERCEPTRON (MLP)
# ==========================================

def build_mlp(input_shape, num_classes):
    model = models.Sequential(name="MLP_NeuralZOO")
    
    # Entrée et mise à plat automatique du volume (32x32x3 -> vecteur de 3072)
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    
    # Couches denses interconnectées
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    
    # Couche de sortie (6 neurones, activation Softmax pour obtenir des probabilités)
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

mlp_model = build_mlp(input_shape, num_classes)


# ==========================================
# 3. MODÈLE 2 : CONVOLUTIONAL NEURAL NETWORK (CNN)
# ==========================================

def build_cnn(input_shape, num_classes):
    model = models.Sequential(name="CNN_NeuralZOO")
    
    # Bloc Convolutif 1 : Détection des features simples (bords, textures)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2))) # Réduction de 32x32 à 16x16
    
    # Bloc Convolutif 2 : Combinaison des features (formes, motifs géométriques)
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2))) # Réduction de 16x16 à 8x8
    
    # Bloc Convolutif 3 : Features complexes (membres, textures d'animaux)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2))) # Réduction de 8x8 à 4x4
    
    # Tête de classification (Fully Connected)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

cnn_model = build_cnn(input_shape, num_classes)


# ==========================================
# 4. COMPILATION DES DEUX MODÈLES
# ==========================================

# Puisque les labels ont été recodés en entiers continus (0 à 5),
# 'sparse_categorical_crossentropy' est la fonction mathématique idéale.
loss_func = 'sparse_categorical_crossentropy'
optimiseur = 'adam'

mlp_model.compile(optimizer=optimiseur, loss=loss_func, metrics=['accuracy'])
cnn_model.compile(optimizer=optimiseur, loss=loss_func, metrics=['accuracy'])

print("\n--- Modèles initialisés et compilés avec succès ! ---")