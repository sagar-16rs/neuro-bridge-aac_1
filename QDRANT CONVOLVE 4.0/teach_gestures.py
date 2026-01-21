import cv2
import time
import uuid
import numpy as np
from vectorizer import NeuroVectorizer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- INIT ---
vectorizer = NeuroVectorizer()

# 1. Connect to the Database
client = QdrantClient(path="qdrant_db_v9") 
COLLECTION_NAME = "gesture_vocabulary"

# 2. ENSURE COLLECTION EXISTS (The Fix)
# This builds the "box" if it doesn't exist yet.
try:
    client.get_collection(COLLECTION_NAME)
except:
    print(f"📦 Creating new collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=99, distance=Distance.COSINE)
    )

def record_gesture(label_name, duration=2):
    cap = cv2.VideoCapture(0)
    print(f"\n🔴 GET READY! Recording '{label_name}' in 3 seconds...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1... ACT NOW!")
    
    start_time = time.time()
    frames_captured = 0
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret: break
        
        # Show video
        cv2.putText(frame, f"Recording: {label_name}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Recording', frame)
        cv2.waitKey(1)
        
        # 3. Vectorize
        pose_vec, _ = vectorizer.get_biometrics(frame)
        
        # 4. Handle Data Type Safely
        if isinstance(pose_vec, list):
            final_vector = pose_vec
        else:
            final_vector = pose_vec.tolist()

        # 5. Save to Memory
        if np.sum(final_vector) != 0:
            client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=final_vector, 
                        payload={"label": label_name}
                    )
                ]
            )
            frames_captured += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved {frames_captured} snapshots for '{label_name}'.")

if __name__ == "__main__":
    print("Commands: Type a gesture name (e.g., 'IDLE', 'WATER') to record.")
    while True:
        choice = input("\nGesture Name (or 'q' to quit): ").strip()
        if choice.lower() == 'q': break
        record_gesture(choice)