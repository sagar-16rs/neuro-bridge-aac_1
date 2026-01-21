import cv2
import time
import uuid
from vectorizer import NeuroVectorizer
from qdrant_client import QdrantClient, models # Added models
from qdrant_client.models import PointStruct

# Connect to Qdrant
client = QdrantClient(path="./neuro_bridge_memory")
vectorizer = NeuroVectorizer()

print("🧹 Preparing Memory...")

# FIXED: Use 'models.VectorParams' and 'models.Distance.EUCLID'
# This prevents spelling errors like "Euclidean" vs "Euclid"
if client.collection_exists("health_baseline"):
    client.delete_collection("health_baseline")

client.create_collection(
    collection_name="health_baseline",
    vectors_config=models.VectorParams(
        size=3, 
        distance=models.Distance.EUCLID
    )
)

print("📷 LOOK AT CAMERA WITH WIDE OPEN EYES (NORMAL STATE)...")
time.sleep(2)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    pose_vec, face_vec = vectorizer.get_biometrics(frame)
    print(f"✅ Captured Baseline: {face_vec}")
    
    # Store this as the "Gold Standard" for this patient
    client.upsert(
        collection_name="health_baseline",
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=face_vec,
                payload={"state": "alert_baseline", "timestamp": time.time()}
            )
        ]
    )
    print("💾 Health Baseline Saved to Qdrant!")

cap.release()