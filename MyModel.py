import cv2
import pickle
import numpy as np
import mediapipe as mp
from typing import List
from scipy.special import softmax

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Added

class MyModel:
    model_path = 'assets/char_model.pkl'
    le_path = 'assets/char_labelencoder.pkl'

    def __init__(self) -> None:
        with open(MyModel.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(MyModel.le_path, 'rb') as f:
            self.le = pickle.load(f)
    
    def predict(self, images: List[np.ndarray]):
        landmarks = np.zeros((len(images), 21, 2))
        with mp_hands.Hands() as hands:
            indices = []
            for i, image in enumerate(images):
                try:
                    # Convert color and process
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                    # Draw the hand landmarks on the image (so you see the pose)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS
                            )
                    
                    # Grab the world landmarks
                    mp_landmarks = results.multi_hand_world_landmarks[0]

                except (TypeError, IndexError):
                    indices.append(i)
                    continue

                # Store the x,y coords
                for j, landmark in enumerate(mp_landmarks.landmark):
                    landmarks[i, j] = landmark.x, landmark.y

        landmarks = landmarks.reshape(len(images), 21*2)
        labelsIdx = self.model.predict(landmarks)
        labelsProp = np.max(softmax(self.model.decision_function(landmarks), 1), 1)
        labels = self.le.inverse_transform(labelsIdx)
        
        # In case we couldn't detect a hand in some images
        labels[indices] = -1
        return labels, labelsProp
