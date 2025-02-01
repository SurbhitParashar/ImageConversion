import cv2
import numpy as np
from py_avataaars import PyAvataaar
avatar=PyAvataaar()
import py_avataaars as pa   
import io
# import tensorflow as tf
# from tensorflow import keras
# load_model = keras.models.load_model
import cv2 as cv
from matplotlib import pyplot as plt
import os
from deepface import DeepFace
from sklearn.cluster import KMeans
from scipy.spatial import distance
from rembg import remove
from PIL import Image
import mediapipe as mp

input_path = "sample9.jpg"
output_path = "refined_image.png"

with open(input_path, "rb") as input_file:
    input_data = input_file.read()

output_data = remove(input_data)

with open(output_path, "wb") as output_file:
    output_file.write(output_data)


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Read the image
image_path_refined = "refined_image.png"  # Update the path with your image name
image = cv2.imread(image_path_refined)

if image is None:
    raise ValueError("Image not found. Please check the image path.")

# Convert the image to RGB for MediaPipe
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces using MediaPipe
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(image_rgb)

if not results.detections:
    raise ValueError("No faces detected in the image.")

# Extract the forehead region from the first detected face
for detection in results.detections:
    # Get bounding box information
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = image.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

    # Define the forehead region (top part of the face)
    forehead_top = max(0, y - h // 4)  # Move slightly upward for better accuracy
    forehead_bottom = y + h // 6  # Use top 1/6th of the face as the forehead
    forehead_left = x + w // 6  # Narrow the forehead region
    forehead_right = x + (5 * w // 6)

    # Crop the forehead region
    forehead = image[forehead_top:forehead_bottom, forehead_left:forehead_right]
    cv2.imwrite("forehead.png", forehead)
    break  # Use the first detected face for the forehead

# Convert the forehead to RGB format for clustering
forehead = cv2.cvtColor(forehead, cv2.COLOR_BGR2RGB)
forehead_pixels = forehead.reshape(-1, 3)
forehead_pixels = np.float32(forehead_pixels)

# Define criteria for K-Means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(forehead_pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Get the dominant skin color
skin_color = centers[0].astype(int)

# Define a color palette
color_palette = {
    'black': (0, 0, 0),
    'brown': (139, 69, 19),
    'dark brown': (101, 67, 33),
    'light': (255, 228, 196),
    'pale': (255, 239, 189),
    'tanned': (210, 180, 140),
    'yellow': (255, 255, 0)
}

# Function to calculate Euclidean distance
def euclidean_distance(color1, color2):
    return np.sqrt((color1[0] - color2[0])**2 + (color1[1] - color2[1])**2 + (color1[2] - color2[2])**2)

# Function to find the closest color
def closest_color(requested_color):
    min_distance = float('inf')
    closest_color_name = None
    for name, color in color_palette.items():
        distance = euclidean_distance(requested_color, color)
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name
    return closest_color_name

# Find the closest color
closest_color_name = closest_color(tuple(skin_color)).replace(" ", "_").upper()

# Map the skin color to pa.SkinColor
skin_color_mapping = {
    "BLACK": pa.SkinColor.BLACK,
    "BROWN": pa.SkinColor.BROWN,
    "DARK_BROWN": pa.SkinColor.DARK_BROWN,
    "LIGHT": pa.SkinColor.LIGHT,
    "PALE": pa.SkinColor.PALE,
    "TANNED": pa.SkinColor.TANNED,
    "YELLOW": pa.SkinColor.YELLOW
}

# Choose the corresponding skin color
chosen_skin_color = skin_color_mapping.get(closest_color_name, pa.SkinColor.LIGHT)

# Output the result
print(f"Detected dominant skin color: {closest_color_name}")
print(f"Mapped skin color to: {chosen_skin_color}")

# hair color code



# Available hair colors in py_avataaars (as RGB)
AVAILABLE_HAIR_COLORS = {
    "Black": (0, 0, 0),
    "Brown_Dark": (101, 67, 33),
    "Brown": (139, 69, 19),
    "Blonde": (250, 240, 190),
    "Auburn": (179, 101, 56),
    "Blonde_Golden": (255, 223, 138),
    "Pastel_Pink": (255, 182, 193),
    "Platinum": (229, 228, 226),
    "Red": (255, 0, 0),
    "Silver_Gray": (192, 192, 192),
}

def find_closest_color(detected_rgb):
    """Find the closest available color based on Euclidean distance."""
    closest_color_name = None
    min_distance = float('inf')

    for color_name, color_rgb in AVAILABLE_HAIR_COLORS.items():
        dist = distance.euclidean(detected_rgb, color_rgb)
        if dist < min_distance:
            min_distance = dist
            closest_color_name = color_name

    return closest_color_name


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Step 1: Load the image
image = cv2.imread(image_path_refined)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Detect face and landmarks using MediaPipe
results = face_mesh.process(image_rgb)

if not results.multi_face_landmarks:
    print("No face detected.")
    exit()

# Step 3: Extract the hair region based on face landmarks
for face_landmarks in results.multi_face_landmarks:
    # Use specific landmarks to define the hair region (above the forehead)
    # Forehead landmarks (e.g., 10, 338, 297, etc.)
    forehead_landmarks = [10, 338, 297, 332, 284]

    # Calculate the bounding box for the hair region
    forehead_points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                       for i, landmark in enumerate(face_landmarks.landmark) if i in forehead_landmarks]
    
    x_coords = [pt[0] for pt in forehead_points]
    y_coords = [pt[1] for pt in forehead_points]

    # Expand the region above the forehead
    hair_region_top = max(0, min(y_coords) - (max(y_coords) - min(y_coords)) // 2)
    hair_region_bottom = min(image.shape[0], min(y_coords))
    hair_region_left = max(0, min(x_coords) - (max(x_coords) - min(x_coords)) // 2)
    hair_region_right = min(image.shape[1], max(x_coords) + (max(x_coords) - min(x_coords)) // 2)

    hair_region = image_rgb[hair_region_top:hair_region_bottom, hair_region_left:hair_region_right]
    cv2.imwrite('hair_region_detected.png', hair_region)

    if hair_region.size == 0:
        print("No hair region detected.")
        exit()

    # Step 4: Preprocess the hair region for color analysis
    hair_pixels = hair_region.reshape((-1, 3))  # Flatten the region into (R, G, B) pixels
    hair_pixels = np.float32(hair_pixels)

    # Step 5: Use K-Means to find the dominant color in the hair region
    k = 1  # Number of dominant colors to detect
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(hair_pixels)

    # Get the dominant color and convert to integer
    dominant_color = kmeans.cluster_centers_[0]
    dominant_color = [int(c) for c in dominant_color]

    # Step 6: Find the closest available color in py_avataaars
    chosen_hair_color = find_closest_color(dominant_color)
    print(f"Closest Hair Color for py_avataaars: {chosen_hair_color}")

# Clean up
face_mesh.close()



def detect_beard_level(image_path_refined):
    # Load the image
    img = cv2.imread(image_path_refined)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return "No face detected"

        # Initialize beard intensity levels
        beard_level = "No Beard"
        h, w, _ = img.shape  # Get image dimensions

        for face_landmarks in results.multi_face_landmarks:
            # Extract coordinates for the lower face region
            lower_face_points = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 324, 91, 181, 146]
            lower_face_coords = []

            for point in lower_face_points:
                x = int(face_landmarks.landmark[point].x * w)
                y = int(face_landmarks.landmark[point].y * h)
                lower_face_coords.append((x, y))

            # Create a mask for the lower face region
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(lower_face_coords, dtype=np.int32)], 255)

            # Extract the lower face region
            lower_face_region = cv2.bitwise_and(img, img, mask=mask)
            gray_lower_face = cv2.cvtColor(lower_face_region, cv2.COLOR_BGR2GRAY)

            # Edge detection to analyze beard density
            edges = cv2.Canny(gray_lower_face, threshold1=50, threshold2=150)

            # Calculate beard density
            beard_density = np.sum(edges) / (np.count_nonzero(mask))
            print("Beard Density:", beard_density)

            # Categorize beard density
            if beard_density < 35.0:
                beard_level = "DEFAULT"
            elif beard_density < 50.0:
                beard_level = "BEARD_LIGHT"
            elif beard_density < 60.0:
                beard_level = "BEARD_MEDIUM"
            else:
                beard_level = "BEARD_HEAVY"

            # Draw a polygon around the lower face region and display beard level
            cv2.polylines(img, [np.array(lower_face_coords, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(img, beard_level, (lower_face_coords[0][0], lower_face_coords[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    

    return beard_level



# Detect beard level
beard_level = detect_beard_level(image_path_refined)
print("Beard Level:", beard_level)

# Beard level mapping
beard_mapping = {
    "DEFAULT": pa.FacialHairType.DEFAULT,
    "BEARD_LIGHT": pa.FacialHairType.BEARD_LIGHT,
    "BEARD_MEDIUM": pa.FacialHairType.BEARD_MEDIUM,
    "BEARD_MAJESTIC": pa.FacialHairType.BEARD_MAJESTIC
}

chosen_beard_level = beard_mapping.get(beard_level, "FacialHairType.DEFAULT")
print("Chosen Beard Level:", chosen_beard_level)


# mouth type code
emotion_to_mouth_type = {
    "happy": "SMILE",
    "sad": "SAD",
    "surprise": "SCREAM_OPEN",
    "neutral": "DEFAULT",
    "angry": "SERIOUS",
    "fear": "DISBELIEF",
    "disgust": "CONCERNED",
}

# Detect emotion and map to mouth type
def detect_mouth_type(image_path_refined):
    try:
        analysis = DeepFace.analyze(img_path=image_path_refined, actions=["emotion"])
        # print("analysis is :",analysis)

        dominant_emotion = analysis[0]["dominant_emotion"]
        return emotion_to_mouth_type.get(dominant_emotion, "DEFAULT")
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
 # Replace with your image path
mouth_type = detect_mouth_type(image_path_refined)
print(f"Detected mouth type: {mouth_type}")

# mapping 
mouth_mapping = {
    "SMILE": pa.MouthType.SMILE,
    "SAD": pa.MouthType.SAD,
    "SCREAM_OPEN": pa.MouthType.SCREAM_OPEN,
    "DEFAULT": pa.MouthType.DEFAULT,
    "SERIOUS": pa.MouthType.SERIOUS,
    "DISBELIEF": pa.MouthType.DISBELIEF,
    "CONCERNED": pa.MouthType.CONCERNED
}

chosen_mouth_type = mouth_mapping.get(mouth_type, pa.MouthType.DEFAULT)
# print("mouth typw",chosen_mouth_type)
# hair type code

import random
from deepface import DeepFace

# Hair style lists for males
SHORT_HAIR_MALE = [
    "SHORT_HAIR_SHORT_CURLY",
    "SHORT_HAIR_SHORT_FLAT",
    "SHORT_HAIR_SHORT_WAVED",
    "SHORT_HAIR_THE_CAESAR",
    "SHORT_HAIR_THE_CAESAR_SIDE_PART",
    "SHORT_HAIR_SHORT_ROUND",
    "SHORT_HAIR_DREADS_01"
]

MEDIUM_HAIR_MALE = [
    "SHORT_HAIR_SHAGGY_MULLET",
    "LONG_HAIR_SHAVED_SIDES",
    "LONG_HAIR_NOT_TOO_LONG"
]

LONG_HAIR_MALE = [
    "LONG_HAIR_BUN"
]

# Hair style lists for females
SHORT_HAIR_FEMALE = [
    "SHORT_HAIR_SHORT_CURLY",
    "SHORT_HAIR_SHORT_FLAT",
]

MEDIUM_HAIR_FEMALE = [
    "LONG_HAIR_STRAIGHT_STRAND",
    "LONG_HAIR_STRAIGHT2",
    "LONG_HAIR_MIA_WALLACE"
]

LONG_HAIR_FEMALE = [
    "LONG_HAIR_CURVY",
    "LONG_HAIR_CURLY",
    "LONG_HAIR_BIG_HAIR",
    "LONG_HAIR_FRIDA"
]

# Mapping of hair styles to topType (using the name of the hairstyle directly)
hair_style_map = {
    "SHORT_HAIR_SHORT_CURLY": pa.TopType.SHORT_HAIR_SHORT_CURLY,
    "SHORT_HAIR_SHORT_FLAT": pa.TopType.SHORT_HAIR_SHORT_FLAT,
    "SHORT_HAIR_SHORT_WAVED": pa.TopType.SHORT_HAIR_SHORT_WAVED,
    "SHORT_HAIR_THE_CAESAR": pa.TopType.SHORT_HAIR_THE_CAESAR,
    "SHORT_HAIR_THE_CAESAR_SIDE_PART": pa.TopType.SHORT_HAIR_THE_CAESAR_SIDE_PART,
    "SHORT_HAIR_SHORT_ROUND": pa.TopType.SHORT_HAIR_SHORT_ROUND,
    "SHORT_HAIR_DREADS_01": pa.TopType.SHORT_HAIR_DREADS_01,
    "SHORT_HAIR_SHAGGY_MULLET": pa.TopType.SHORT_HAIR_SHAGGY_MULLET,
    "LONG_HAIR_SHAVED_SIDES": pa.TopType.LONG_HAIR_SHAVED_SIDES,
    "LONG_HAIR_NOT_TOO_LONG": pa.TopType.LONG_HAIR_NOT_TOO_LONG,
    "LONG_HAIR_CURVY": pa.TopType.LONG_HAIR_CURVY,
    "LONG_HAIR_BUN": pa.TopType.LONG_HAIR_BUN,
    "LONG_HAIR_STRAIGHT_STRAND": pa.TopType.LONG_HAIR_STRAIGHT_STRAND,
    "LONG_HAIR_STRAIGHT2": pa.TopType.LONG_HAIR_STRAIGHT2,
    "LONG_HAIR_MIA_WALLACE": pa.TopType.LONG_HAIR_MIA_WALLACE,
    "LONG_HAIR_CURLY": pa.TopType.LONG_HAIR_CURLY,
    "LONG_HAIR_BIG_HAIR": pa.TopType.LONG_HAIR_BIG_HAIR,
    "LONG_HAIR_FRIDA": pa.TopType.LONG_HAIR_FRIDA,

}

def detect_gender(image_path_refined):
    """Detect gender using DeepFace."""
    try:
        # Perform gender analysis
        analysis = DeepFace.analyze(img_path=image_path_refined, actions=['gender'], enforce_detection=False)
        
        # If analysis returns a list, take the first detected face
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        # Extract gender probabilities
        if 'gender' in analysis:
            gender_probabilities = analysis['gender']
            print(gender_probabilities)
            # Find the gender with the highest probability
            if gender_probabilities['Man'] > gender_probabilities['Woman']:
                return "Man"
            else:
                return "Woman"
        else:
            return "Error: Gender not found in analysis."
    except Exception as e:
        return f"Error detecting gender: {str(e)}"

def detect_hair_style(image_path_refined, gender):
    """Detect hair style based on edge density and gender."""
    try:
        # Load the image
        image = cv2.imread(image_path_refined)
        if image is None:
            return "Error: Unable to read the image. Please check the file path.", None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Focus on the upper region (potential hair area)
        height, width = edges.shape
        hair_region = edges[:int(height * 0.4), :]  # Top 40% of the image
        
        # Analyze hair density in the region
        hair_density = np.sum(hair_region) / (hair_region.shape[0] * hair_region.shape[1])
        print(f"Hair Density: {hair_density}")  # Debugging

        # Select appropriate hair styles based on gender and hair density
        if gender == "Man":
           
            if hair_density < 10.0:
                hair_style = random.choice(SHORT_HAIR_MALE)
            elif 10.0 <= hair_density < 15.0:
                hair_style = random.choice(MEDIUM_HAIR_MALE)
            else:
                hair_style = random.choice(LONG_HAIR_MALE)
        elif gender == "Woman":
            
            if hair_density < 1.0:
                hair_style = random.choice(SHORT_HAIR_FEMALE)
            elif 1.0 <= hair_density < 1.5:
                hair_style = random.choice(MEDIUM_HAIR_FEMALE)
            else:
                hair_style = random.choice(LONG_HAIR_FEMALE)
        else:
            return "Gender not detected. Cannot determine hair style.", None

        print(f"Detected Hair Style: {hair_style}")  # Debugging

        # Get the corresponding topType from the map
        topType = hair_style_map.get(hair_style)
        if topType is None:
            print(f"Error: Hair style '{hair_style}' not found in the map.")  # Debugging
            topType = "Unknown topType"

        return hair_style, topType

    except Exception as e:
        return f"Error detecting hair style: {str(e)}", None

# To use the functions
def analyze_image(image_path_refined):
    gender = detect_gender(image_path_refined)
    # print("Gender:", gender)
    if "Error" not in gender:
        hair_style, top_type = detect_hair_style(image_path_refined, gender)
        # print("here",hair_style)
        return gender, hair_style, top_type
    return gender, None, None

# Example usage
gender_result, hair_style_result, chosen_top_type = analyze_image(image_path_refined)

print(f"Detected Gender: {gender_result}")
print(f"Detected Hair Style: {hair_style_result}")
print(f"Mapped Top Type: {chosen_top_type}")


# choosing eye type 
eye_types = ["DEFAULT", "HAPPY", "SIDE"]
selected_eye_type_name = random.choice(eye_types)

eye_type_mapping = {
    "DEFAULT": pa.EyesType.DEFAULT,
    "HAPPY": pa.EyesType.HAPPY,
    "SIDE": pa.EyesType.SIDE
}

chosen_eye_type = eye_type_mapping[selected_eye_type_name]
# print(chosen_eye_type)

# eyebrow type
 # Make sure this is the correct import for `pa.EyebrowType`

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

def analyze_expression_and_map_eyebrows(image_path_refined):
    try:
        # Analyze facial expression using DeepFace
        analysis = DeepFace.analyze(img_path=image_path_refined, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        # print(f"Detected emotion: {emotion}")

        # Map the detected emotion to an eyebrow type
        eyebrow_key = emotion_to_eyebrow_type.get(emotion, "DEFAULT_NATURAL")
        return eyebrow_type_mapping.get(eyebrow_key, pa.EyebrowType.DEFAULT_NATURAL)
    
    except Exception as e:
        print(f"Error in analyzing the image: {e}")
        return pa.EyebrowType.DEFAULT_NATURAL  # Return a default eyebrow type in case of error

# Load an image and detect facial expression
  # Replace with your image path
chosen_eyebrow_type = analyze_expression_and_map_eyebrows(image_path_refined)

accesories_choice=["DEFAULT",
"PRESCRIPTION_01",
"PRESCRIPTION_02",
"ROUND",
"SUNGLASSES",
"WAYFARERS"
]
accesories_type=random.choice(accesories_choice)

accesories_type_map={
    "DEFAULT":pa.AccessoriesType.DEFAULT,
    "KURT":pa.AccessoriesType.KURT,
    "PRESCRIPTION_01":pa.AccessoriesType.PRESCRIPTION_01,
    "PRESCRIPTION_02":pa.AccessoriesType.PRESCRIPTION_02,
    "ROUND":pa.AccessoriesType.ROUND,
    "SUNGLASSES":pa.AccessoriesType.SUNGLASSES,
    "WAYFARERS":pa.AccessoriesType.WAYFARERS
}

chosen_accesories_type=accesories_type_map.get(accesories_type,pa.AccessoriesType.DEFAULT)

# cloth type
cloth_types=["HOODIE,"
"BLAZER_SHIRT",
"BLAZER_SWEATER",
"COLLAR_SWEATER",
"GRAPHIC_SHIRT",
"OVERALL",
"SHIRT_V_NECK"]
cloth_type=random.choice(cloth_types)
cloth_mapping={
    "HOODIE":pa.ClotheType.HOODIE,
    "BLAZER_SHIRT":pa.ClotheType.BLAZER_SHIRT,
    "BLAZER_SWEATER":pa.ClotheType.BLAZER_SWEATER,
    "COLLAR_SWEATER":pa.ClotheType.COLLAR_SWEATER,
    "GRAPHIC_SHIRT":pa.ClotheType.GRAPHIC_SHIRT,
    "OVERALL":pa.ClotheType.OVERALL,
    "SHIRT_V_NECK":pa.ClotheType.SHIRT_V_NECK
}
chosen_cloth_type=cloth_mapping.get(cloth_type,pa.ClotheType.HOODIE)

print(chosen_skin_color)
print(chosen_beard_level)
print(chosen_hair_color)
print(chosen_top_type)
print(chosen_mouth_type)
print(chosen_eye_type)
print(chosen_eyebrow_type)
print(chosen_accesories_type)
print(chosen_cloth_type)


# Create an avatar instance
avatar = pa.PyAvataaar(
    skin_color=chosen_skin_color,
    facial_hair_type=chosen_beard_level,
    hair_color=chosen_hair_color, 
    top_type=chosen_top_type, #hair type
    mouth_type=chosen_mouth_type,
    eye_type=chosen_eye_type,
    eyebrow_type=chosen_eyebrow_type,
    nose_type=pa.NoseType.DEFAULT,
    accessories_type=chosen_accesories_type,
    clothe_type=chosen_cloth_type,
    # clothe_graphic_type=pa.ClotheGraphicType.PIZZA,
)

# Render and save the avatar as a PNG file
avatar.render_png_file("avt2.png")


svg_code=avatar.render_svg_file("avt2.svg")

