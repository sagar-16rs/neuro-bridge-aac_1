# db_manager.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class QdrantBrain:
    def __init__(self):
        # --- FIX: USING LOCAL MODE (No Docker needed) ---
        # This saves the database to a folder in your project directory
        self.client = QdrantClient(path="./neuro_bridge_memory")
        self._setup_collections()

    def _setup_collections(self):
        # 1. Gesture Vocabulary
        if not self.client.collection_exists("gesture_vocabulary"):
            self.client.create_collection(
                collection_name="gesture_vocabulary",
                vectors_config=VectorParams(size=99, distance=Distance.COSINE)
            )
            print("✅ Memory Created: Gesture Vocabulary")
        else:
            print("ℹ️  Memory Found: Gesture Vocabulary")

        # 2. Patient Baseline
        if not self.client.collection_exists("patient_baseline"):
            self.client.create_collection(
                collection_name="patient_baseline",
                vectors_config=VectorParams(size=1404, distance=Distance.EUCLID)
            )
            print("✅ Memory Created: Patient Baseline")
        else:
             print("ℹ️  Memory Found: Patient Baseline")

        # 3. Interaction Logs
        if not self.client.collection_exists("interaction_logs"):
            self.client.create_collection(
                collection_name="interaction_logs",
                vectors_config=VectorParams(size=99, distance=Distance.COSINE)
            )
            print("✅ Memory Created: Interaction Logs")
        else:
             print("ℹ️  Memory Found: Interaction Logs")

if __name__ == "__main__":
    brain = QdrantBrain()
    print("🧠 Neuro-Bridge System Initialized (Local Mode).")