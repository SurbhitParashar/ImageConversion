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

input_path = "image_male.png"
output_path = "refined_image.png"

with open(input_path, "rb") as input_file:
    input_data = input_file.read()

output_data = remove(input_data)

with open(output_path, "wb") as output_file:
    output_file.write(output_data)


# Read the image
image_path_refined = "refined_image.png"  # Update the path with your image name
image = cv2.imread(image_path_refined)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

if len(faces) == 0:
    raise ValueError("No faces detected in the image.")

# Extract the forehead region
for (x, y, w, h) in faces:
    # Ensure the forehead region is within bounds
    reduction_factor=0.2
    left_reduction=int(w*reduction_factor)
    right_reduction=int(w*reduction_factor)
    forehead = image[y:y + h // 4, x+left_reduction:x + w-right_reduction]
    # cv2.imwrite("forehead.png",forehead)
    # print(forehead)
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


skin_color_mapping = {
    "BLACK": pa.SkinColor.BLACK,
    "BROWN": pa.SkinColor.BROWN,
    "DARK_BROWN": pa.SkinColor.DARK_BROWN,
    "LIGHT": pa.SkinColor.LIGHT,
    "PALE": pa.SkinColor.PALE,
    "TANNED": pa.SkinColor.TANNED,
    "YELLOW": pa.SkinColor.YELLOW
}

chosen_skin_color = skin_color_mapping.get(closest_color_name, pa.SkinColor.LIGHT) 

# avatar.render_png_file("avt.png")

# hair color code


# Available hair colors in py_avataaars (as RGB)
AVAILABLE_HAIR_COLORS = {
    "Black": (0, 0, 0),                 # Black
    "Brown_Dark": (101, 67, 33),        # Dark Brown
    "Brown": (139, 69, 19),             # Brown
    "Blonde": (250, 240, 190),          # Blonde
    "Auburn": (179, 101, 56),           # Auburn (reddish brown)
    "Blonde_Golden": (255, 223, 138),   # Blonde Golden (light golden blonde)
    "Pastel_Pink": (255, 182, 193),     # Pastel Pink
    "Platinum": (229, 228, 226),        # Platinum Blonde (light grayish blonde)
    "Red": (255, 0, 0),                 # Red
    "Silver_Gray": (192, 192, 192),     # Silver Gray
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


# Step 1: Load the image
 # Update with your image path
image = cv2.imread(image_path_refined)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Detect face in the image using Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

if len(faces) == 0:
    print("No face detected.")
    exit()

# Step 3: Extract the hair region (above the face)
for (x, y, w, h) in faces:
    # Define the hair region: an area above the face
    hair_region_top = max(0, y - h // 2)
    hair_region = image_rgb[hair_region_top:y, x:x+w]
    # cv.imwrite('hairs.png', hair_region)

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
closest_hair_color = find_closest_color(dominant_color)
print(f"Closest Hair Color for py_avataaars: {closest_hair_color}")

hair_color_mapping = {
    "Black": pa.HairColor.BLACK,
    "Brown_Dark": pa.HairColor.BROWN_DARK,
    "Brown": pa.HairColor.BROWN,
    "Blonde": pa.HairColor.BLONDE,
    "Auburn": pa.HairColor.AUBURN,
    "Blonde_Golden": pa.HairColor.BLONDE_GOLDEN,
    "Pastel_Pink": pa.HairColor.PASTEL_PINK,
    "Platinum": pa.HairColor.PLATINUM,
    "Red": pa.HairColor.RED,
    "Silver_Gray": pa.HairColor.SILVER_GRAY
}

chosen_hair_color = hair_color_mapping.get(closest_hair_color, pa.HairColor.BLACK)


def detect_beard_level(image_path_refined):
    # Load pre-trained face detection model (Haar cascades or DNN model)
    
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    img = cv2.imread(image_path_refined)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return "No face detected"

    # Initialize beard intensity levels
    beard_level = "No Beard"

    for (x, y, w, h) in faces:
        shift_ratio = int(w * 0.5)  # Adjust the value between 0 and 1 for the amount of shift (e.g., 20%)
        shift = int(w * 0.2)
        shift_pixels = int(w * shift_ratio)

        # Crop the face region
        face_region = gray[y:y + h, x - shift:x + shift_ratio]
        # face_region = gray[y:y + h, x:x + w] basic
        # cv.imwrite("face_Region.png", face_region)
        # Focus on lower half of the face (where beard typically is)
        lower_face_region = face_region[h // 3:h:, :]
        # cv.imwrite("lower_face.png", lower_face_region)
        # Use edge detection to highlight beard intensity
        edges = cv2.Canny(lower_face_region, threshold1=50, threshold2=150)

        # Calculate beard density
        beard_density = np.sum(edges) / (lower_face_region.size)
        print("Beard Density:", beard_density)

        # Categorize beard density
        if beard_density < 37.0:
            beard_level = "DEFAULT"
        elif beard_density < 60.0:
            beard_level = "BEARD_LIGHT"
        elif beard_density < 70.0:
            beard_level = "BEARD_MEDIUM"
        else:
            beard_level = "BEARD_HEAVY"

        # Draw rectangle around face and display beard level
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, beard_level, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


    return beard_level


# Input image path


# Detect beard level
beard_level = detect_beard_level(image_path_refined)
# print("Beard Level:", beard_level)

# mapping 
beard_mapping = {
    "DEFAULT": pa.FacialHairType.DEFAULT,
    "BEARD_LIGHT": pa.FacialHairType.BEARD_LIGHT,
    "BEARD_MEDIUM": pa.FacialHairType.BEARD_MEDIUM,
    "BEARD_MAJESTIC": pa.FacialHairType.BEARD_MAJESTIC
}
 
chosen_beard_level = beard_mapping.get(beard_level, pa.FacialHairType.DEFAULT)

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
    "LONG_HAIR_FRIDA",
    "LONG_HAIR_DREADS"
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
    "LONG_HAIR_DREADS": pa.TopType.LONG_HAIR_DREADS,

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

# print(chosen_skin_color)
# print(chosen_beard_level)
# print(chosen_hair_color)
# print(chosen_top_type)
# print(chosen_mouth_type)
# print(chosen_eye_type)
# print(chosen_eyebrow_type)
# print(chosen_accesories_type)



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
    # clothe_shirt_type=pa.ClotheShirtType.DEFAULT,
    # clothe_type=pa.clothe_type.GRAPHIC_SHIRT,
    # clothe_color=pa.clothe_color.PINK,
    # clothe_graphic_type=pa.clothe_graphic_type.BIRD,
    # clothe_graphic_color=pa.clothe_graphic_color.WHITE
)

# Render and save the avatar as a PNG file
avatar.render_png_file("avt2.png")


svg_code=avatar.render_svg_file("avt2.svg")

