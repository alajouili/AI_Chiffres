import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# 1. Chargement du modèle entraîné
print("Chargement du modèle...")
model = tf.keras.models.load_model('mnist.h5')

# 2. Configuration de MediaPipe (Détection de la main)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 3. Création du "Canvas" (Tableau noir virtuel pour dessiner)
canvas = np.zeros((480, 640, 1), dtype=np.uint8) # Image noire
prev_x, prev_y = 0, 0 # Pour mémoriser la position précédente du doigt

# Lancement de la webcam
cap = cv2.VideoCapture(0)

def preprocess_image(img):
    """Prépare le dessin pour que l'IA puisse le lire (zoom + centrage)"""
    # Trouver les pixels blancs (le dessin)
    coords = cv2.findNonZero(img)
    if coords is None: return None # Si rien n'est dessiné
    
    # Découper un rectangle autour du dessin (Bounding Box)
    x, y, w, h = cv2.boundingRect(coords)
    
    # Ajouter une petite marge autour
    padding = 15
    roi = img[max(0, y-padding):min(img.shape[0], y+h+padding), 
              max(0, x-padding):min(img.shape[1], x+w+padding)]
    
    if roi.size == 0: return None

    # Redimensionner en 28x28 comme le modèle MNIST l'attend
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = roi / 255.0 # Normaliser entre 0 et 1
    roi = roi.reshape(1, 28, 28, 1) # Mettre au format (1 image, 28, 28, 1 canal)
    return roi

print("Lancement de la caméra... (Appuie sur 'q' pour quitter)")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Miroir horizontal (plus naturel)
    frame = cv2.flip(frame, 1)
    
    # Détection de la main
    f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(f_rgb)
    
    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            # Récupérer la position de l'index (Point n°8)
            lm = hand_lms.landmark[8]
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            
            # Dessiner un cercle vert sur l'index
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            
            # Si c'est la première détection, on initialise la position
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy
            
            # DESSINER : Trait blanc sur le canvas noir
            cv2.line(canvas, (prev_x, prev_y), (cx, cy), (255), 25)
            # DESSINER : Trait rouge sur l'écran (pour que tu voies)
            cv2.line(frame, (prev_x, prev_y), (cx, cy), (0, 0, 255), 5)
            
            prev_x, prev_y = cx, cy
    else:
        prev_x, prev_y = 0, 0 # Reset si on perd la main

    # 4. Prédiction de l'IA
    digit_img = preprocess_image(canvas)
    prediction_text = "?"
    
    if digit_img is not None:
        pred = model.predict(digit_img, verbose=0)
        class_id = np.argmax(pred) # Le chiffre avec la plus haute probabilité
        confidence = np.max(pred)  # Le % de certitude
        
        if confidence > 0.6: # Si l'IA est sûre à plus de 60%
            prediction_text = str(class_id)
            
    # 5. Affichage
    # Incrustation de ce que l'IA "voit" en haut à gauche
    canvas_mini = cv2.resize(canvas, (150, 150))
    canvas_mini = cv2.cvtColor(canvas_mini, cv2.COLOR_GRAY2BGR)
    frame[0:150, 0:150] = canvas_mini
    
    # Affichage du résultat
    cv2.putText(frame, f"Prediction: {prediction_text}", (10, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    cv2.putText(frame, "Effacer: 'c' | Quitter: 'q'", (10, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("IA Chiffres - Air Writing", frame)
    
    # Contrôles clavier
    key = cv2.waitKey(1)
    if key == ord('q'): break
    if key == ord('c'): canvas = np.zeros((480, 640, 1), dtype=np.uint8) # Effacer

cap.release()
cv2.destroyAllWindows()