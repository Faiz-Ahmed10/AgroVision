# Model paths and configurations
MODEL_CONFIG = {
    'IMG_SIZE': (224, 224),
    'DISEASE_MODEL_PATH': "models/final.keras",
    'CROP_MODEL_PATH': "models/gaussiannb_crop_model.pkl",
    'LABEL_ENCODER_PATH': "models/label_encoder.pkl",
    'FERTILIZER_MODEL_PATH': 'models/model.pkl',
    'SCALER_PATH': 'models/scaler.pkl'
}

# Class names for disease prediction
CLASS_NAMES = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy", 
    "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight",
    "Corn Healthy", "Grape Black Rot", "Grape Esca (Black Measles)",
    "Grape Leaf Blight", "Grape Healthy", "Orange Haunglongbing (Citrus Greening)",
    "Peach Bacterial Spot", "Peach Healthy", "Pepper Bell Bacterial Spot", 
    "Pepper Bell Healthy", "Potato Early Blight", "Potato Late Blight", 
    "Potato Healthy", "Raspberry Healthy", "Soybean Healthy", 
    "Squash Powdery Mildew", "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", 
    "Tomato Mosaic Virus", "Tomato Healthy"
]

# Language codes for translation
LANGUAGE_CODES = {
    'hindi': 'hi',
    'urdu': 'ur',
    'telugu': 'te'
}

# Fertilizer names mapping
FERTILIZER_NAMES = {
    0: "Urea",
    1: "DAP",
    2: "MOP",
    3: "Complex",
    4: "SSP",
    5: "Ammonium Sulphate",
    6: "Gypsum"
}

# Emoji mappings
EMOJI_MAPPING = {
    'N': ('🌿', 'N (Nitrogen ratio)'),
    'P': ('⚡', 'P (Phosphorous ratio)'),
    'K': ('💧', 'K (Potassium ratio)'),
    'temperature': ('🌡️', '°C (Temperature in Celsius)'),
    'humidity': ('💧', '% (Humidity)'),
    'ph': ('🔬', 'pH (Soil pH value)'),
    'rainfall': ('🌧️', 'mm (Rainfall in mm)')
}

CROP_EMOJI_MAPPING = {
    'Rice': '🌾',
    'Maize (Corn)': '🌽',
    'Chickpea': '🫘',
    'Kidney Beans': '🫘',
    'Pigeon Peas': '🫘',
    'Moth Beans': '🫘',
    'Mung Bean': '🫘',
    'Black Gram': '🫘',
    'Lentil': '🫘',
    'Pomegranate': '🍎',
    'Banana': '🍌',
    'Mango': '🥭',
    'Grapes': '🍇',
    'Watermelon': '🍉',
    'Muskmelon': '🍈',
    'Apple': '🍎',
    'Orange': '🍊',
    'Papaya': '🥭',
    'Coconut': '🥥',
    'Cotton': '🌱',
    'Jute': '🪢',
    'Coffee': '☕'
}