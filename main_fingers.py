import cv2
import mediapipe as mp

# 1. Configuration de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Les IDs des bouts des doigts (Index, Majeur, Annulaire, Auriculaire)
finger_tips_ids = [8, 12, 16, 20]

def count_fingers(lm_list):
    """Compte le nombre de doigts levés basé sur les coordonnées"""
    fingers = []

    # 1. Le Pouce (Cas spécial car il bouge sur le côté)
    # On compare la position X du bout (4) par rapport à l'articulation (3)
    # Note : Cette logique suppose la main droite face caméra (ou gauche en miroir)
    if lm_list[4][0] > lm_list[3][0]: 
        fingers.append(1) # Pouce levé (ouvert)
    else:
        fingers.append(0) # Pouce fermé

    # 2. Les 4 autres doigts (Index à Auriculaire)
    # On compare la position Y (hauteur). Attention : en image, Y=0 est tout en haut.
    # Donc si le bout (tip) est < à l'articulation (pip), le doigt est levé.
    for tip_id in finger_tips_ids:
        # On compare le bout du doigt (tip_id) avec l'articulation 2 points plus bas (tip_id - 2)
        if lm_list[tip_id][1] < lm_list[tip_id - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    # Somme des 1 pour avoir le total
    return fingers.count(1)

print("Lancement... Montre tes doigts à la caméra !")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Miroir pour que ce soit plus naturel
    frame = cv2.flip(frame, 1)
    
    # Conversion RGB pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    number_text = "0"
    
    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            # Dessiner les points rouges sur la main
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Récupérer les coordonnées de tous les points (0 à 20)
            lm_list = []
            h, w, c = frame.shape
            for id, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([cx, cy])
            
            # Appeler notre fonction de comptage
            if len(lm_list) != 0:
                total_fingers = count_fingers(lm_list)
                number_text = str(total_fingers)
                
                # Affichage conditionnel (Juste pour le fun)
                msg = ""
                if total_fingers == 0: msg = "Poing ferme"
                elif total_fingers == 2: msg = "Peace !"
                elif total_fingers == 5: msg = "Bonjour !"
                
                cv2.putText(frame, msg, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Affichage du Gros Chiffre
    # Fond vert pour le chiffre
    cv2.rectangle(frame, (10, 10), (150, 100), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, number_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

    cv2.imshow("Detection Doigts", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()