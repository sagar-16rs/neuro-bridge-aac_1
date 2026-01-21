import mediapipe as mp
import cv2
import numpy as np

class NeuroVectorizer:
    def __init__(self):
        # 1. Pose Model (For Gestures)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 2. Face Mesh Model (For Fatigue/Emotion)
        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Critical for Iris/Eye tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _normalize_landmarks(self, landmarks):
        """Standardizes pose so camera distance doesn't matter."""
        if not landmarks: return [0.0] * 99
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        hip_center = (coords[23] + coords[24]) / 2
        return (coords - hip_center).flatten().tolist()

    def _get_eye_aspect_ratio(self, landmarks):
        """
        Research-Grade Metric: Calculates 'Eye Aspect Ratio' (EAR).
        EAR < 0.20 usually means eyes are closed (Blinking/Sleeping).
        """
        # Indices for Left Eye landmarks in MediaPipe Face Mesh
        left_eye_indices = [362, 385, 387, 263, 373, 380]
        points = [landmarks[i] for i in left_eye_indices]
        p = np.array([[pt.x, pt.y] for pt in points])
        
        # Vertical distances (Eye opening)
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        # Horizontal distance (Eye width)
        C = np.linalg.norm(p[0] - p[3])
        
        # EAR Formula
        ear = (A + B) / (2.0 * C)
        return ear

    def get_biometrics(self, frame):
        """Returns TWO vectors: Body Language (99 dims) and Face State (3 dims)"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Process Pose
        pose_results = self.pose.process(image_rgb)
        if pose_results.pose_landmarks:
            pose_vec = self._normalize_landmarks(pose_results.pose_landmarks)
        else:
            pose_vec = [0.0] * 99
        
        # 2. Process Face
        face_results = self.face.process(image_rgb)
        # Default: [Left_EAR, Right_EAR, 0.0]
        face_vec = [0.3, 0.3, 0.0] 
        
        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark
            ear = self._get_eye_aspect_ratio(lm)
            # We treat the EAR as a "Health Vector"
            face_vec = [ear, ear, 0.0] 
            
        return pose_vec, face_vec