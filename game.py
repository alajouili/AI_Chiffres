import cv2
import mediapipe as mp
import random
import time

# --- CONFIGURATION ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Largeur HD
cap.set(4, 720)  # Hauteur HD

finger_tips_ids = [8, 12, 16, 20]

# --- VARIABLES DU JEU ---
score = 0
target_number = random.randint(1, 10) # Le premier nombre à deviner
start_time = time.time()
message = f"Montre le chiffre: {target_number}"
background_color = (50, 50, 50) # Gris foncé

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Conversion et détection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    total_fingers = 0
    
    # --- LOGIQUE DE COMPTAGE (La même qu'avant) ---
    if result.multi_hand_landmarks:
        for hand_index, hand_lms in enumerate(result.multi_hand_landmarks):
            # Dessin plus joli : Lignes vertes épaisses
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
            
            lm_list = []
            for lm in hand_lms.landmark:
                lm_list.append([int(lm.x * w), int(lm.y * h)])
            
            # GESTION GAUCHE / DROITE
            hand_label = result.multi_handedness[hand_index].classification[0].label
            
            # Pouce
            if hand_label == "Right":
                if lm_list[4][0] < lm_list[3][0]: total_fingers += 1
            else:
                if lm_list[4][0] > lm_list[3][0]: total_fingers += 1
            
            # Autres doigts
            for tip_id in finger_tips_ids:
                if lm_list[tip_id][1] < lm_list[tip_id - 2][1]:
                    total_fingers += 1

    # --- LOGIQUE DU JEU ---
    
    # Barre d'info en haut
    cv2.rectangle(frame, (0, 0), (1280, 80), background_color, cv2.FILLED)
    
    # Si le joueur a bon
    if total_fingers == target_number:
        score += 1
        # Effet visuel : Écran vert flash
        cv2.rectangle(frame, (0, 0), (1280, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, "BRAVO !", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
        cv2.imshow("Jeu Math Hands", frame)
        cv2.waitKey(500) # Pause de 0.5 sec pour savourer la victoire
        
        # Choisir un nouveau nombre (différent du précédent)
        new_target = random.randint(1, 10)
        while new_target == target_number:
            new_target = random.randint(1, 10)
        target_number = new_target
    
    # Affichages Textes
    # 1. Le Score à gauche
    cv2.putText(frame, f"Score: {score}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    
    # 2. La cible au milieu (Ce qu'il faut faire)
    text_target = f"Montre: {target_number}"
    cv2.putText(frame, text_target, (450, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # 3. Ce que l'IA voit (ton nombre actuel) en bas
    cv2.putText(frame, f"Detecte: {total_fingers}", (20, 650), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    cv2.imshow("Jeu Math Hands", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()