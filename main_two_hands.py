import cv2
import mediapipe as mp
import math

# --- CONFIGURATION ---
mp_hands = mp.solutions.hands
# On autorise 2 mains maintenant pour aller jusqu'à 10
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# IDs des bouts des doigts (sauf le pouce)
finger_tips_ids = [8, 12, 16, 20]

def get_hand_label(index, hand_results, w, h):
    """Récupère si c'est la main Gauche ou Droite"""
    output = None
    for idx, classification in enumerate(hand_results.multi_handedness):
        if classification.classification[0].index == index:
            # MediaPipe inverse souvent G/D en mode selfie, on récupère le label
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = "{} {}".format(label, round(score, 2))
            
            # Récupérer les coordonnées du poignet pour afficher le label
            coords = tuple(np.multiply(
                np.array((hand_results.multi_hand_landmarks[index].landmark[mp_hands.HandLandmark.WRIST].x, 
                          hand_results.multi_hand_landmarks[index].landmark[mp_hands.HandLandmark.WRIST].y)),
            [w, h]).astype(int))
            
            output = label
    return output

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Miroir
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Conversion RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    total_fingers = 0
    
    if result.multi_hand_landmarks:
        # Pour chaque main détectée
        for hand_index, hand_lms in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Liste des coordonnées
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([cx, cy])
            
            # --- LOGIQUE INTELLIGENTE GAUCHE / DROITE ---
            # On demande à MediaPipe quelle main c'est (Left ou Right)
            # Note : En mode miroir (flip), "Right" est ta main droite réelle.
            hand_label = result.multi_handedness[hand_index].classification[0].label
            
            # Compter les doigts de CETTE main
            fingers_on_hand = 0
            
            # 1. Le Pouce (Logique corrigée)
            # Si main DROITE : Pouce est à GAUCHE de l'articulation (car x augmente vers la droite)
            if hand_label == "Right":
                if lm_list[4][0] < lm_list[3][0]: # < car pouce vers l'intérieur
                    fingers_on_hand += 1
            # Si main GAUCHE
            else:
                if lm_list[4][0] > lm_list[3][0]: # > car pouce vers l'intérieur inverse
                    fingers_on_hand += 1
            
            # 2. Les 4 autres doigts (Toujours la hauteur Y)
            for tip_id in finger_tips_ids:
                if lm_list[tip_id][1] < lm_list[tip_id - 2][1]:
                    fingers_on_hand += 1
            
            total_fingers += fingers_on_hand

    # --- AFFICHAGE ---
    # Affichage du chiffre total (0 à 10)
    # Rectangle vert dynamique : Change de couleur si 0
    color = (0, 255, 0) # Vert
    if total_fingers == 0: color = (0, 0, 255) # Rouge si 0
    
    cv2.rectangle(frame, (20, 20), (170, 170), color, cv2.FILLED)
    cv2.putText(frame, str(total_fingers), (45, 145), cv2.FONT_HERSHEY_SIMPLEX, 
                5, (255, 255, 255), 10)
    
    cv2.imshow("Compteur 0-10", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()