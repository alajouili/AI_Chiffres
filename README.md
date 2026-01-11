# 🎮 AI Chiffres - Jeu de Reconnaissance Intelligente

Bienvenue sur le dépôt de **AI Chiffres** ! Ce projet est une application interactive basée sur l'Intelligence Artificielle qui permet de jouer avec la reconnaissance de chiffres et la détection de gestes.

## 🚀 Fonctionnalités

* **Reconnaissance de chiffres manuscrits :** Utilise un modèle de Deep Learning (CNN entraîné sur MNIST) pour deviner les chiffres que vous dessinez.
* **Détection des mains :** Utilise la vision par ordinateur pour compter le nombre de doigts levés ou reconnaître des gestes.
* **Interface Web :** Une interface fluide et facile à utiliser grâce à Streamlit.
* **Système de jeu :** Différents niveaux de difficulté pour tester votre rapidité et précision.

## 🛠️ Installation

Pour lancer ce projet sur votre machine, suivez ces étapes :

1.  **Cloner le projet :**
    ```bash
    git clone [https://github.com/alajouili/AI_Chiffres.git](https://github.com/alajouili/AI_Chiffres.git)
    cd AI_Chiffres
    ```

2.  **Installer les dépendances :**
    Assurez-vous d'avoir Python installé, puis lancez :
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Comment jouer ?

Une fois les installations terminées, lancez l'application avec la commande suivante dans votre terminal :

```bash
streamlit run app.py
Le jeu s'ouvrira automatiquement dans votre navigateur web par défaut.

📂 Structure du projet
app.py : Le fichier principal qui lance l'interface Streamlit.

mnist.h5 : Le modèle d'intelligence artificielle entraîné pour reconnaître les chiffres.

game.py : La logique du jeu.

main_fingers.py & main_two_hands.py : Scripts de détection des mains.

👤 Auteur
Projet réalisé par alajouili.
