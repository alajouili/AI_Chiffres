import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import random
import time

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Math Hands Arcade", 
    page_icon="🖐️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS PERSONNALISÉ (LE DESIGN) ---
st.markdown("""
    <style>
    /* Fond d'écran sombre avec dégradé */
    .stApp {
        background: linear-gradient(to bottom right, #0f0c29, #302b63, #24243e);
        color: white;
    }
    
    /* Titre Néon */
    h1 {
        text-align: center;
        font-family: 'Courier New', Courier, monospace;
        color: #00ffcc;
        text-shadow: 0 0 10px #00ffcc, 0 0 20px #00ffcc, 0 0 40px #00ffcc;
        margin-bottom: 30px;
    }

    /* Style du HUD (Score et Vies) */
    .hud-container {
        display: flex;
        justify-content: space-around;
        background-color: rgba(0, 0, 0, 0.6);
        padding: 15px;
        border-radius: 15px;
        border: 2px solid #00ffcc;
        margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.3);
    }
    
    .hud-item {
        font-size: 24px;
        font-weight: bold;
        font-family: sans-serif;
    }

    /* Boutons et Sidebar */
    .stButton>button {
        width: 100%;
        background-color: #ff0055;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3377;
        box-shadow: 0 0 10px #ff0055;
    }
    
    /* Cacher le menu hamburger par défaut pour faire "App" */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. INIT MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
finger_tips_ids = [8, 12, 16, 20]

# --- 4. MOTEUR DU JEU ---
def run_game(difficulty_speed):
    # Création des zones d'affichage (Placeholders)
    hud_placeholder = st.empty()     # Pour le Score/Vies
    video_placeholder = st.empty()   # Pour la Vidéo
    
    cap = cv2.VideoCapture(0)
    score = 0
    lives = 3
    falling_numbers = []
    spawn_timer = 0
    game_active = True
    
    while cap.isOpened() and game_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Impossible d'accéder à la caméra.")
            break
            
        # Miroir + Dimensions
        frame = cv2.flip(frame, 1)
        h_screen, w_screen, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- Détection ---
        result = hands.process(rgb_frame)
        total_fingers = 0
        
        if result.multi_hand_landmarks:
            for hand_index, hand_lms in enumerate(result.multi_hand_landmarks):
                # Dessin "Cyber" (Vert fluo et blanc)
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0, 255, 204), thickness=2, circle_radius=4),
                                       mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
                
                # Logique de comptage (inchangée)
                lm_list = []
                for lm in hand_lms.landmark:
                    lm_list.append([int(lm.x * w_screen), int(lm.y * h_screen)])
                
                hand_label = result.multi_handedness[hand_index].classification[0].label
                if hand_label == "Right":
                    if lm_list[4][0] < lm_list[3][0]: total_fingers += 1
                else:
                    if lm_list[4][0] > lm_list[3][0]: total_fingers += 1
                for tip_id in finger_tips_ids:
                    if lm_list[tip_id][1] < lm_list[tip_id - 2][1]: total_fingers += 1

        # --- Gestion des Ennemis ---
        current_time = time.time()
        spawn_rate = 2.5 - (difficulty_speed * 0.2)
        
        if current_time - spawn_timer > spawn_rate:
            spawn_timer = current_time
            falling_numbers.append({
                'x': random.randint(50, w_screen - 50),
                'y': 0,
                'val': random.randint(1, 10),
                'speed': random.randint(3 + difficulty_speed, 8 + difficulty_speed)
            })

        for enemy in falling_numbers[:]:
            enemy['y'] += enemy['speed']
            
            # Hit (Succès)
            if total_fingers == enemy['val']:
                score += 1
                falling_numbers.remove(enemy)
                # Effet visuel : Cercle vert qui s'agrandit
                cv2.circle(frame, (enemy['x'], enemy['y']), 70, (0, 255, 100), 5)
                continue
            
            # Miss (Échec)
            if enemy['y'] > h_screen:
                lives -= 1
                falling_numbers.remove(enemy)
                if lives <= 0:
                    game_active = False
            
            # Dessin de la bulle ennemie (Style Arcade)
            # Cercle extérieur
            cv2.circle(frame, (enemy['x'], enemy['y']), 45, (255, 0, 100), cv2.FILLED)
            # Cercle intérieur
            cv2.circle(frame, (enemy['x'], enemy['y']), 35, (255, 100, 150), cv2.FILLED)
            # Chiffre
            cv2.putText(frame, str(enemy['val']), (enemy['x']-15, enemy['y']+15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # --- Affichage Interface ---
        
        # 1. Mise à jour du HUD HTML (Le bandeau score)
        heart_icon = "❤️" * lives
        hud_html = f"""
        <div class="hud-container">
            <div class="hud-item">🎯 Score: <span style="color:#00ffcc">{score}</span></div>
            <div class="hud-item">🖐️ Détecté: <span style="color:yellow">{total_fingers}</span></div>
            <div class="hud-item">{heart_icon}</div>
        </div>
        """
        hud_placeholder.markdown(hud_html, unsafe_allow_html=True)

        # 2. Mise à jour de la vidéo
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

    # --- FIN DE PARTIE ---
    cap.release()
    video_placeholder.empty()
    hud_placeholder.empty()
    
    st.markdown(f"""
        <div style="text-align: center; margin-top: 50px;">
            <h1 style="color: #ff0055; font-size: 80px; text-shadow: 0 0 20px red;">GAME OVER</h1>
            <h2 style="color: white;">Score Final : {score}</h2>
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; display: inline-block;">
                Appuie sur <b>"Arrêter"</b> puis <b>"Lancer"</b> à gauche pour rejouer !
            </div>
        </div>
    """, unsafe_allow_html=True)


# --- 5. STRUCTURE DE LA PAGE ---

# Titre Principal
st.title("👾 NEON MATH HANDS 👾")
st.markdown("*Utilise tes mains pour détruire les chiffres avant qu'ils ne touchent le sol !*")

# Barre latérale (Contrôles)
with st.sidebar:
    st.markdown("## 🎮 Contrôles")
    st.markdown("---")
    
    difficulty = st.slider("Niveau de Difficulté", 1, 10, 3)
    
    st.write(" ") # Espacement
    
    # Bouton Start/Stop stylisé
    start_game = st.checkbox("🔥 LANCER LE JEU", value=False)
    
    st.markdown("---")
    st.info("💡 **Astuce :** Assure-toi d'avoir un bon éclairage pour que la caméra voie bien tes doigts.")

# Lancement
if start_game:
    run_game(difficulty)
else:
    # Écran d'accueil quand le jeu ne tourne pas
    st.markdown("""
    <div style="text-align: center; padding: 50px; background: rgba(0,0,0,0.3); border-radius: 20px;">
        <h3>👈 Coche la case 'Lancer le Jeu' pour commencer</h3>
        <p>Prépare tes deux mains !</p>
    </div>
    """, unsafe_allow_html=True)