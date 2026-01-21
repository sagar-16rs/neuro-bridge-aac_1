import time
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class NeuroMemory:
    def __init__(self):
        # Initialize Local Qdrant (Persistent Storage)
        self.client = QdrantClient(path="qdrant_db_v9") 
        self.collection_name = "gesture_vocabulary"
        self.log_collection = "interaction_logs"
        
        # Ensure collections exist (Size 99 for face mesh)
        self._init_collection(self.collection_name, 99) 
        self._init_collection(self.log_collection, 99) 

    def _init_collection(self, name, size):
        try:
            self.client.get_collection(name)
        except:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE)
            )

    # ------------------------------------------------------------------
    # FIX: Added this function to handle the command from main_agent
    # ------------------------------------------------------------------
    def store_experience(self, vector, label, fatigue_level=None):
        """
        Compatibility layer: Routes 'store_experience' calls to 'evolve_memory'
        """
        print(f"📥 Storing experience: {label} (Fatigue: {fatigue_level})")
        self.evolve_memory(vector, label)

    def retrieve_command(self, vector, fatigue_score):
        """
        RESEARCH FEATURE: Fatigue-Adaptive Retrieval
        """
        adaptive_threshold = 0.88 - (fatigue_score * 0.20)
        
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector.tolist(),
                limit=1,
                score_threshold=0.0 
            )
        except AttributeError:
            print("⚠️ Warning: Using fallback search method.")
            search_result = self.client.search_batch(
                collection_name=self.collection_name,
                requests=[{"vector": vector.tolist(), "limit": 1}]
            )[0]

        response = {
            "match": False,
            "command": "Unknown",
            "confidence": 0.0,
            "threshold_used": adaptive_threshold
        }
        
        if search_result:
            top_hit = search_result[0]
            response["confidence"] = top_hit.score
            response["command"] = top_hit.payload.get("label", "Unknown")
            
            if top_hit.score > adaptive_threshold:
                response["match"] = True
                
        return response

    def evolve_memory(self, vector, label):
        """
        ACTIVE LEARNING: Upserting a 'Negotiated' vector as Gold Standard.
        """
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "label": label,
                        "type": "evolved",
                        "timestamp": time.time()
                    }
                )
            ]
        )
        print(f"💾 MEMORY EVOLVED: {label} (ID: {point_id})")

    def log_interaction(self, vector, command, confidence, action, fatigue):
        """
        DATA COLLECTION: Logs every frame.
        """
        self.client.upsert(
            collection_name=self.log_collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={
                        "predicted_cmd": command,
                        "confidence": confidence,
                        "action_taken": action, 
                        "fatigue_level": fatigue,
                        "timestamp": time.time()
                    }
                )
            ]
        )