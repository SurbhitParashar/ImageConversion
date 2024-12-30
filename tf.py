from deepface import DeepFace
import cv2
import py_avataaars as pa  # Make sure this is the correct import for `pa.EyebrowType`

# Define eyebrow mapping based on emotions
emotion_to_eyebrow_type = {
    "neutral": "DEFAULT_NATURAL",
    "happy": "RAISED_EXCITED",
    "sad": "FROWN_NATURAL",
    "angry": "UP_DOWN_NATURAL",
    "surprise": "RAISED_EXCITED",
    "fear": "UP_DOWN",
    "disgust": "FLAT_NATURAL",
}

# Define the corresponding `pa.EyebrowType` mapping
eyebrow_type_mapping = {
    "DEFAULT_NATURAL": pa.EyebrowType.DEFAULT_NATURAL,
    "RAISED_EXCITED": pa.EyebrowType.RAISED_EXCITED,
    "FROWN_NATURAL": pa.EyebrowType.FROWN_NATURAL,
    "UP_DOWN_NATURAL": pa.EyebrowType.UP_DOWN_NATURAL,
    "UP_DOWN": pa.EyebrowType.UP_DOWN,
    "FLAT_NATURAL": pa.EyebrowType.FLAT_NATURAL
}

def analyze_expression_and_map_eyebrows(image_path):
    try:
        # Analyze facial expression using DeepFace
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        print(f"Detected emotion: {emotion}")

        # Map the detected emotion to an eyebrow type
        eyebrow_key = emotion_to_eyebrow_type.get(emotion, "DEFAULT_NATURAL")
        return eyebrow_type_mapping.get(eyebrow_key, pa.EyebrowType.DEFAULT_NATURAL)
    
    except Exception as e:
        print(f"Error in analyzing the image: {e}")
        return pa.EyebrowType.DEFAULT_NATURAL  # Return a default eyebrow type in case of error

# Load an image and detect facial expression
image_path = "refined_image.png"  # Replace with your image path
chosen_eyebrow_type = analyze_expression_and_map_eyebrows(image_path)


