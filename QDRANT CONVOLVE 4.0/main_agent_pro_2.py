import cv2
import numpy as np
import time
import pyttsx3
import threading
from collections import deque
from vectorizer import NeuroVectorizer
from memory_agent import NeuroMemory

# --- 1. THE ACTUATOR (Voice Engine) ---
class VoiceActuator:
    def __init__(self):
        self.lock = threading.Lock()
        
    def speak(self, text, urgent=False):
        """Non-blocking, thread-safe speech."""
        t = threading.Thread(target=self._run_speech, args=(text, urgent))
        t.daemon = True
        t.start()
        
    def _run_speech(self, text, urgent):
        with self.lock:
            try:
                engine = pyttsx3.init()
                rate = 165 if urgent else 140
                engine.setProperty('rate', rate)
                engine.say(text)
                engine.runAndWait()
            except: pass

# --- 2. THE REASONING ENGINE (Symbolic State Machine) ---
class CognitiveAgent:
    def __init__(self):
        self.state = "IDLE" # IDLE, NEGOTIATING, EXECUTING
        self.pending_intent = None
        self.pending_vector = None
        self.negotiation_timer = 0
        
    def process_signals(self, visual_intent, visual_conf, blink_detected):
        """
        Decides: Do we trust the neural net, or do we negotiate?
        """
        response = {
            "action": "WAIT", 
            "ui_text": "Scanning...", 
            "color": (0,255,0)
        }
        
        CONFIDENCE_THRESHOLD_AMBIGUOUS = 0.45 
        
        # --- STATE: NEGOTIATING (Waiting for Blink) ---
        if self.state == "NEGOTIATING":
            response["ui_text"] = f"CONFIRM: {self.pending_intent}?"
            response["color"] = (0, 165, 255) # Orange
            
            # Timeout (2.5s) -> Reject
            if time.time() - self.negotiation_timer > 2.5:
                self.state = "IDLE"
                response["ui_text"] = "Confirmation Timed Out"
                response["action"] = "TIMEOUT"
                return response
            
            # Feedback Received (Blink) -> Learn & Execute
            if blink_detected:
                response["action"] = "LEARN_AND_EXECUTE"
                response["intent"] = self.pending_intent
                response["vector"] = self.pending_vector
                self.state = "IDLE"
                return response

        # --- STATE: IDLE (Normal Scanning) ---
        else:
            # 1. Strong Match (Neural Confidence High)
            if visual_conf > 0.88 or (visual_intent != "Unknown" and visual_conf > 0.80):
                self.state = "EXECUTING"
                response["action"] = "EXECUTE"
                response["intent"] = visual_intent
                response["ui_text"] = f"DETECTED: {visual_intent}" # UPGRADE: Update HUD text
                
            # 2. Ambiguous Match (Neural Confidence Medium)
            elif visual_conf > CONFIDENCE_THRESHOLD_AMBIGUOUS and visual_intent != "Unknown":
                self.state = "NEGOTIATING"
                self.pending_intent = visual_intent
                self.pending_vector = None # Will be filled in main loop
                self.negotiation_timer = time.time()
                response["action"] = "ASK_CONFIRMATION"
                response["intent"] = visual_intent
        
        return response

# --- 3. UI VISUALIZATION ---
fatigue_history = []
def draw_sci_fi_hud(img, fatigue_score, state_text, blinks, log_count):
    h, w, _ = img.shape
    
    # Fatigue Graph
    fatigue_history.append(fatigue_score)
    if len(fatigue_history) > 100: fatigue_history.pop(0)
    for i in range(1, len(fatigue_history)):
        val_1 = int(fatigue_history[i-1] * 100)
        val_2 = int(fatigue_history[i] * 100)
        cv2.line(img, (20 + (i-1)*2, h-40-val_1), (20 + i*2, h-40-val_2), (0,255,255), 1)
    
    cv2.putText(img, "FATIGUE METRIC", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

    # Agent State Box
    color = (0,255,0) if "CONFIRM" not in state_text else (0,165,255)
    cv2.rectangle(img, (w-250, 20), (w-20, 100), (20,20,20), -1)
    cv2.rectangle(img, (w-250, 20), (w-20, 100), color, 1)
    cv2.putText(img, "NEURO-SYMBOLIC STATE", (w-240, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv2.putText(img, state_text[:18], (w-240, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Data Log Counter
    cv2.putText(img, f"LOGS: {log_count}", (w-100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
    
    # Blink Indicator
    if blinks:
        cv2.circle(img, (w-30, 40), 8, (0,0,255), -1)

# --- 4. MAIN SYSTEM ---
def main():
    # Initialize Modules
    cortex = NeuroMemory() 
    senses = NeuroVectorizer() 
    agent = CognitiveAgent()
    voice = VoiceActuator()
    
    cap = cv2.VideoCapture(0)
    
    # State Variables
    jitter_buffer = [] 
    eyes_closed = False
    last_spoken_time = 0
    recent_blink_detected = False
    log_counter = 0
    
    BLINK_THRESHOLD = 0.22

    print("🧠 Neuro-Bridge 9.0 (Research Edition) ONLINE.")
    print("📂 Logging data to Qdrant/interaction_logs for analysis.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- A. PERCEPTION LAYER ---
        try:
            raw_pose_vec, face_vec = senses.get_biometrics(frame)
        except: continue
        
        if np.sum(raw_pose_vec) == 0: continue

        # Stabilization
        jitter_buffer.append(raw_pose_vec)
        if len(jitter_buffer) > 5: jitter_buffer.pop(0)
        stable_vec = np.mean(np.array(jitter_buffer), axis=0)
        
        # Fatigue Calculation
        current_ear = face_vec[0]
        fatigue_score = max(0, 0.30 - current_ear)
        
        # Blink Detection
        recent_blink_detected = False
        if current_ear < BLINK_THRESHOLD:
            if not eyes_closed: eyes_closed = True
        else:
            if eyes_closed: # Eye opened -> Blink complete
                eyes_closed = False
                recent_blink_detected = True

        # --- B. MEMORY RETRIEVAL (Adaptive) ---
        memory_result = cortex.retrieve_command(stable_vec, fatigue_score)
        
        # --- C. REASONING LAYER (The Agent) ---
        agent.pending_vector = stable_vec 
        
        decision = agent.process_signals(
            visual_intent = memory_result['command'],
            visual_conf = memory_result['confidence'],
            blink_detected = recent_blink_detected
        )
        
        # --- D. EXECUTION & LOGGING ---
        
        # 1. EXECUTE (Strong Match)
        if decision["action"] == "EXECUTE":
            # UPGRADE: SILENCE FILTER FOR IDLE
            intent_clean = decision['intent'].lower()
            if intent_clean in ["idle", "neutral", "nothing"]:
                # Do nothing (Stay Silent), just update HUD
                pass 
            else:
                # Only speak for real commands (Water, Pain, etc.)
                if time.time() - last_spoken_time > 4.0:
                    phrase = f"I need {decision['intent']}"
                    if decision['intent'] == "PAIN": phrase = "I am in Pain."
                    
                    print(f"🤖 AGENT ACT: {phrase}")
                    voice.speak(phrase)
                    last_spoken_time = time.time()
                    
                    # Log meaningful interactions
                    cortex.log_interaction(stable_vec, decision['intent'], memory_result['confidence'], "EXECUTE", fatigue_score)
                    log_counter += 1
            
        # 2. ASK (Ambiguity)
        elif decision["action"] == "ASK_CONFIRMATION":
            if time.time() - last_spoken_time > 4.0: 
                voice.speak(f"Did you mean {decision['intent']}?")
                last_spoken_time = time.time()
                
                cortex.log_interaction(stable_vec, decision['intent'], memory_result['confidence'], "NEGOTIATE", fatigue_score)
                log_counter += 1
            
        # 3. LEARN (Feedback Loop Closed)
        elif decision["action"] == "LEARN_AND_EXECUTE":
            print(f"🧬 EVOLVING: Saving {decision['intent']}")
            cortex.evolve_memory(decision['vector'], decision['intent'])
            voice.speak("Updating memory.")
            time.sleep(1.0)
            voice.speak(f"I need {decision['intent']}")
            last_spoken_time = time.time()
            cortex.log_interaction(stable_vec, decision['intent'], 1.0, "EVOLUTION", fatigue_score)
            log_counter += 1

        # --- E. UI DRAWING ---
        draw_sci_fi_hud(frame, fatigue_score, decision["ui_text"], recent_blink_detected, log_counter)
        
        cv2.imshow('Neuro-Bridge 9.0', frame)
        
        # Inputs
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('1'): cortex.store_experience(stable_vec, "PAIN", "Low") 
        elif k == ord('2'): cortex.store_experience(stable_vec, "WATER", "Low")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()