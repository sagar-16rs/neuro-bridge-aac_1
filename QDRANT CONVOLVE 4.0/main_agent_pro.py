import cv2
import numpy as np
import time
import pyttsx3
import threading
from vectorizer import NeuroVectorizer
from memory_agent import NeuroMemory

# --- SYSTEM INIT ---
cortex = NeuroMemory() 
senses = NeuroVectorizer() 

# --- SIMPLIFIED VOICE ENGINE ---
def speak_immediate(text):
    """Creates a fresh engine instance for every sentence. Reliable."""
    try:
        # Initialize locally to avoid threading conflicts
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) # Speed
        engine.setProperty('volume', 1.0) # Max volume
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Voice Error: {e}")

def speak(text):
    # Run in a separate thread so video doesn't freeze
    t = threading.Thread(target=speak_immediate, args=(text,))
    t.daemon = True # Kills thread if program closes
    t.start()

# --- BIOMETRICS ---
jitter_buffer = [] 
BUFFER_SIZE = 5

def get_stabilized_vector(new_vector):
    jitter_buffer.append(new_vector)
    if len(jitter_buffer) > BUFFER_SIZE:
        jitter_buffer.pop(0)
    stacked = np.array(jitter_buffer)
    return np.mean(stacked, axis=0)

def main():
    cap = cv2.VideoCapture(0)
    training_mode = False
    current_label = "Unknown"
    
    last_spoken_time = 0 
    COOLDOWN = 3.5 # Seconds to wait before speaking again
    
    print("🧠 Neuro-Bridge 6.2 (Simple Voice) ONLINE.")
    print("🔊 Make sure your volume is UP.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        try:
            raw_pose_vec, face_vec = senses.get_biometrics(frame)
        except: continue
        
        if np.sum(raw_pose_vec) == 0: continue

        stable_vec = get_stabilized_vector(raw_pose_vec)
        current_ear = face_vec[0]
        fatigue_score = max(0, 0.30 - current_ear) 
        fatigue_state = "High" if fatigue_score > 0.05 else "Low"

        # DECISION AGENT
        ui_color = (0, 255, 0)
        ui_text = "Scanning..."
        evidence_text = ""

        result = cortex.retrieve_command(stable_vec, fatigue_score)

        if training_mode:
            ui_color = (0, 165, 255)
            ui_text = f"TRAINING: {current_label}"
        
        elif result['match']:
            # GREEN: Success
            command = result['command']
            ui_text = f"DETECTED: {command}"
            evidence_text = result['evidence']
            
            # --- VOICE TRIGGER ---
            # Strict Cooldown Only. If Green, Speak.
            if time.time() - last_spoken_time > COOLDOWN:
                
                phrase = ""
                if command == "WATER":
                    phrase = "Water, please."
                elif command == "PAIN":
                    phrase = "I am in Pain."
                else:
                    phrase = command
                
                print(f"🔊 SPEAKING: {phrase}") # Debug print to console
                speak(phrase)
                last_spoken_time = time.time()

            if result['confidence'] > 0.96:
                 cortex.evolve_memory(stable_vec, command)
                 
        elif result['confidence'] > 0.4:
            ui_text = f"? {result['command']} ({result['confidence']:.2f})"
            ui_color = (100, 100, 100)
            evidence_text = result['evidence']

        # UI
        cv2.rectangle(frame, (0,0), (640, 120), (30,30,30), -1)
        cv2.putText(frame, f"FATIGUE: {fatigue_state} ({fatigue_score:.3f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(frame, f"MODE: {'TRAINING' if training_mode else 'INFERENCE'}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_color, 2)
        cv2.putText(frame, ui_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ui_color, 2)
        
        # Speaking Indicator
        if time.time() - last_spoken_time < 2.0:
             cv2.putText(frame, "🔊", (580, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,255), 3)

        cv2.imshow('Neuro-Bridge 6.2 (Simple Voice)', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('t'): training_mode = not training_mode
        elif k == ord('1'): 
            current_label = "PAIN"
            if training_mode: cortex.store_experience(stable_vec, "PAIN", fatigue_state)
        elif k == ord('2'): 
            current_label = "WATER"
            if training_mode: cortex.store_experience(stable_vec, "WATER", fatigue_state)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()