import cv2
import time
import uuid
from vectorizer import NeuroVectorizer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- SETUP ---
client = QdrantClient(path="./neuro_bridge_memory")
vectorizer = NeuroVectorizer()

# Ensure collection exists
if not client.collection_exists("gesture_vocabulary"):
    client.create_collection(
        collection_name="gesture_vocabulary",
        vectors_config=VectorParams(size=99, distance=Distance.EUCLID)
    )

def record_gesture(label):
    print(f"🎥 PREPARING TO RECORD: '{label}'")
    print("   -> Get ready in 3 seconds...")
    time.sleep(1)
    print("   -> 2...")
    time.sleep(1)
    print("   -> 1... POSE NOW!")
    time.sleep(1)
    
    # Capture 5 frames to get a good average
    cap = cv2.VideoCapture(0)
    vectors = []
    
    for i in range(10): # Capture 10 frames quickly
        ret, frame = cap.read()
        if ret:
            pose_vec, _ = vectorizer.get_biometrics(frame)
            # Only keep valid poses (not empty zeros)
            if sum(pose_vec) != 0:
                vectors.append(pose_vec)
            cv2.imshow('Training...', frame)
            cv2.waitKey(50)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(vectors) > 0:
        # Save ALL valid variations to Qdrant (Better accuracy)
        points = []
        for vec in vectors:
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"meaning": label, "type": "training_sample"}
            ))
            
        client.upsert(collection_name="gesture_vocabulary", points=points)
        print(f"✅ SUCCESS: Saved {len(points)} variations for '{label}'!")
    else:
        print("❌ FAILED: No skeleton detected. Try again.")

# --- MENU LOOP ---
while True:
    print("\n--- NEURO-BRIDGE TRAINER ---")
    print("Type a gesture name to record (e.g., 'Water', 'Cold', 'Bathroom')")
    print("Or type 'exit' to quit.")
    
    label = input(">> ")
    if label.lower() == 'exit':
        break
    
    record_gesture(label)