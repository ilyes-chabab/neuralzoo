# NeuralZoo 

Bienvenue sur le repository du projet **NeuralZoo**. Ce projet a pour objectif d'explorer et de comparer les performances du **Perceptron Multicouche (MLP)** et du **Réseau de Neurones Convolutifs (CNN)** pour la classification d'images d'animaux issues du dataset **CIFAR-10**.

---

##  Veille Technologique

### 1. Perceptron Multicouche (MLP)
Le MLP est une architecture de réseau de neurones "dense" où chaque neurone d'une couche est connecté à tous les neurones de la couche suivante. 
* **Architecture :** Composée d'une couche d'entrée, de couches cachées (couches denses) et d'une couche de sortie.
* **Hyperparamètres clés :** Nombre de couches cachées, nombre de neurones par couche, taux d'apprentissage (*learning rate*), fonctions d'activation.
* **Usage :** Idéal pour les données tabulaires, mais limité pour les images car il ne préserve pas la structure spatiale des pixels.

### 2. Réseau de Neurones Convolutifs (CNN)
Le CNN est l'architecture de référence pour la vision par ordinateur. 
* **Architecture typique :** Elle alterne des **couches de convolution** (qui extraient des caractéristiques locales via des filtres) et des **couches de pooling** (qui réduisent la dimension spatiale), terminant par une couche dense pour la classification.
* **Pourquoi pour les images ?** Contrairement au MLP, le CNN capture des dépendances spatiales locales (textures, formes), ce qui lui permet d'apprendre des hiérarchies de motifs visuels.

---

##  Méthodologie et Application

### Le Dataset : CIFAR-10
Le dataset contient 60 000 images couleur de 32x32 pixels, réparties en 10 classes distinctes.
* **Répartition :** 50 000 images d'entraînement et 10 000 images de test.
* **Objectif :** Classification d'images d'animaux.

### Étapes de développement
1. **Analyse exploratoire :** Nettoyage et filtrage des données.
2. **Modélisation :** Construction d'un modèle MLP et d'un modèle CNN avec `Keras`.
3. **Optimisation :** Utilisation de la *Batch Normalization* et du *Dropout* pour contrer l'overfitting.
4. **Évaluation :** Analyse via matrice de confusion et rapports de classification.

---

##  Résultats et Conclusion
*Le modèle **CNN** s'est révélé être le plus adapté à la problématique de classification d'images grâce à sa capacité à traiter la structure spatiale des données.*

---

##  Installation

```bash
# Clonez le repository
git clone [https://github.com/ilyes-chabab/neuralzoo.git](https://github.com/ilyes-chabab/neuralzoo.git)

# Installez les dépendances
pip install -r requirements.txt
```

##  Architecture du Projet

```text
├── data/              # Dossier contenant les jeux de données
├── models/            # Fichiers de sauvegarde des modèles (.pkl)
├── notebooks  
├── src                # contient les classes
├── tests              # tests
└── requirements.txt   # Dépendances du projet
```