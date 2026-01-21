import cv2
import time
import numpy as np
import uuid
from vectorizer import NeuroVectorizer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# --- CONFIGURATION ---
client = QdrantClient(path="./neuro_bridge_memory")
vectorizer = NeuroVectorizer()

# Research Parameters
BASE_CONFIDENCE = 0.85
FATIGUE_DRIFT_LIMIT = 0.05 

def get_health_baseline():
    try:
        results = client.scroll(collection_name="health_baseline", limit=1)
        if results and results[0]: return results[0][0].vector
    except: pass
    return [0.3, 0.3, 0.0] # Safety Default

def main():
    cap = cv2.VideoCapture(0)
    baseline_face = get_health_baseline()
    
    # 3-Agent State
    health_state = "OK"
    audio_state = "SILENT" # Simulated Audio Agent
    comm_state = "IDLE"
    
    print("🧠 Neuro-Bridge 3.0 (Research Edition) ONLINE.")
    print("ℹ️  Controls: Press 'c' to simulate COUGH/DISTRESS sound.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- 1. SENSING AGENT (Inputs) ---
        pose_vec, face_vec = vectorizer.get_biometrics(frame)
        
        # Simulated Audio Input (Press 'c' to trigger cough)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            audio_state = "HIGH_ENERGY (COUGH)"
            audio_timer = 20 # Keep active for 20 frames
        elif k == ord('q'):
            break
        else:
            # simple timer decay for audio simulation
            if 'audio_timer' in locals() and audio_timer > 0:
                audio_timer -= 1
            else:
                audio_state = "SILENT"

        # --- 2. HEALTH AGENT (Biometric Drift) ---
        current_ear = face_vec[0]
        drift = baseline_face[0] - current_ear
        
        # DYNAMIC THRESHOLDING (The Research Innovation)
        # If drift is high (patient tired), lower the confidence threshold
        current_threshold = BASE_CONFIDENCE
        if drift > FATIGUE_DRIFT_LIMIT:
            health_state = "FATIGUE (DRIFT)"
            current_threshold = 0.70 # Be more forgiving
            health_color = (0, 0, 255)
        else:
            health_state = "STABLE"
            health_color = (0, 255, 0)

        # --- 3. COMMUNICATION AGENT (Retrieval & Evolution) ---
        match_text = "..."
        rec_text = ""
        
        if np.sum(pose_vec) != 0:
            try:
                # SEARCH
                results = client.query_points(
                    collection_name="gesture_vocabulary",
                    query=pose_vec,
                    limit=1
                )
                
                if results.points:
                    best = results.points[0]
                    score = best.score
                    
                    if score > current_threshold:
                        meaning = best.payload['meaning']
                        match_text = f"{meaning} ({int(score*100)}%)"
                        
                        # EVOLVING MEMORY (The "Qdrant" Innovation)
                        # If match is VERY strong (>95%), update memory to track drift
                        if score > 0.96:
                            client.upsert(
                                collection_name="gesture_vocabulary",
                                points=[PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector=pose_vec,
                                    payload={"meaning": meaning, "type": "evolved"}
                                )]
                            )
                            print(f"🧠 Evolving: Learned new nuance of '{meaning}'")

                        # --- 4. CAREGIVER AGENT (Synthesis) ---
                        # Combine Audio + Health + Gesture
                        if audio_state == "HIGH_ENERGY (COUGH)" and meaning == "Pain":
                            rec_text = "⚠️ EMERGENCY: CHOKING/DISTRESS DETECTED"
                        elif health_state == "FATIGUE (DRIFT)":
                            rec_text = f"Suggest Rest. Interpreted '{meaning}' (Low Conf)."
                        else:
                            rec_text = f"Action: {meaning}"
                            
            except Exception as e:
                print(e)

        # --- VISUALIZATION (Research Dashboard) ---
        # 1. Info Panel
        cv2.rectangle(frame, (0,0), (640, 100), (20,20,20), -1)
        
        # Health & Drift
        cv2.putText(frame, f"HEALTH: {health_state} | Drift: {drift:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, health_color, 1)
        
        # Audio
        audio_col = (0, 0, 255) if "HIGH" in audio_state else (100, 100, 100)
        cv2.putText(frame, f"AUDIO: {audio_state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, audio_col, 1)
        
        # Adaptive Threshold Display
        cv2.putText(frame, f"ADAPTIVE THRESHOLD: {current_threshold}", (350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Main Output
        cv2.putText(frame, match_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Recommendation Banner
        if rec_text:
            cv2.rectangle(frame, (0, 400), (640, 480), (0,0,150) if "EMERGENCY" in rec_text else (0,100,0), -1)
            cv2.putText(frame, rec_text, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow('Neuro-Bridge 3.0 (Research Edition)', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()