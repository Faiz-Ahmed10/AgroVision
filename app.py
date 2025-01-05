from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from googletrans import Translator
import json
import numpy as np
import joblib
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)
translator = Translator()

# Load the trained .keras model
MODEL_PATH = "models/final.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define input image size and class names
IMG_SIZE = (224, 224)
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

DISEASE_PRECAUTIONS = {
    "Apple Scab": [
        "Remove and destroy fallen infected leaves",
        "Prune trees to improve air circulation",
        "Apply fungicides early in the growing season",
        "Plant disease-resistant apple varieties"
    ],
    "Apple Black Rot": [
        "Remove mummified fruits from trees and ground",
        "Prune out dead or diseased wood",
        "Maintain good sanitation in the orchard",
        "Apply appropriate fungicides during growing season"
    ],
    "Apple Cedar Rust": [
        "Remove nearby cedar trees if possible",
        "Apply preventive fungicides in spring",
        "Choose resistant apple varieties",
        "Maintain proper tree spacing for ventilation"
    ],
    "Apple Healthy": [
        "Regular pruning for good air circulation",
        "Maintain proper fertilization schedule",
        "Regular monitoring for early disease detection",
        "Practice proper watering techniques"
    ],
    "Blueberry Healthy": [
        "Maintain soil pH between 4.5 and 5.5",
        "Provide adequate mulching",
        "Ensure proper irrigation",
        "Regular pruning for plant health"
    ],
    "Cherry Powdery Mildew": [
        "Improve air circulation through pruning",
        "Apply fungicides at first sign of disease",
        "Avoid overhead irrigation",
        "Remove infected plant parts"
    ],
    "Cherry Healthy": [
        "Regular pruning maintenance",
        "Proper fertilization program",
        "Adequate spacing between trees",
        "Monitor for pest and disease issues"
    ],
    "Corn Cercospora Leaf Spot": [
        "Rotate crops with non-host plants",
        "Remove crop debris after harvest",
        "Use resistant hybrids when available",
        "Apply fungicides when necessary"
    ],
    "Corn Common Rust": [
        "Plant resistant corn varieties",
        "Early planting to avoid optimal rust conditions",
        "Monitor fields regularly",
        "Apply appropriate fungicides if needed"
    ],
    "Corn Northern Leaf Blight": [
        "Rotate with non-host crops",
        "Plant resistant varieties",
        "Remove or plow under crop debris",
        "Apply fungicides at first sign of disease"
    ],
    "Corn Healthy": [
        "Maintain proper plant spacing",
        "Follow recommended fertilization",
        "Regular monitoring for diseases",
        "Proper irrigation management"
    ],
    "Grape Black Rot": [
        "Remove mummified berries and infected leaves",
        "Improve air circulation through pruning",
        "Apply protective fungicides",
        "Maintain proper canopy management"
    ],
    "Grape Esca (Black Measles)": [
        "Remove and destroy infected vines",
        "Avoid pruning during wet weather",
        "Protect pruning wounds",
        "Maintain vine vigor through proper nutrition"
    ],
    "Grape Leaf Blight": [
        "Improve air circulation in vineyard",
        "Remove infected plant material",
        "Apply appropriate fungicides",
        "Avoid overhead irrigation"
    ],
    "Grape Healthy": [
        "Regular pruning and training",
        "Proper fertilization schedule",
        "Maintain good air circulation",
        "Monitor for early disease signs"
    ],
    "Orange Haunglongbing (Citrus Greening)": [
        "Control psyllid populations",
        "Remove infected trees immediately",
        "Use disease-free nursery stock",
        "Regular monitoring for symptoms"
    ],
    "Peach Bacterial Spot": [
        "Use copper-based sprays preventively",
        "Prune during dry weather",
        "Avoid overhead irrigation",
        "Plant resistant varieties"
    ],
    "Peach Healthy": [
        "Regular pruning maintenance",
        "Proper fertilization program",
        "Monitor for pest issues",
        "Maintain good orchard sanitation"
    ],
    "Pepper Bell Bacterial Spot": [
        "Rotate crops with non-hosts",
        "Use disease-free seeds",
        "Avoid working with wet plants",
        "Apply copper-based products preventively"
    ],
    "Pepper Bell Healthy": [
        "Maintain proper spacing",
        "Regular fertilization",
        "Monitor for pest issues",
        "Proper irrigation practices"
    ],
    "Potato Early Blight": [
        "Practice crop rotation",
        "Remove infected plant debris",
        "Apply fungicides preventively",
        "Maintain proper plant spacing"
    ],
    "Potato Late Blight": [
        "Monitor weather conditions",
        "Apply fungicides before disease onset",
        "Destroy volunteer potato plants",
        "Plant resistant varieties"
    ],
    "Potato Healthy": [
        "Proper hilling of plants",
        "Regular fertilization",
        "Monitor for disease symptoms",
        "Maintain proper irrigation"
    ],
    "Raspberry Healthy": [
        "Regular pruning of old canes",
        "Maintain good air circulation",
        "Proper fertilization program",
        "Monitor for pest issues"
    ],
    "Soybean Healthy": [
        "Proper seed spacing",
        "Regular crop monitoring",
        "Maintain proper soil fertility",
        "Practice crop rotation"
    ],
    "Squash Powdery Mildew": [
        "Plant resistant varieties",
        "Improve air circulation",
        "Apply fungicides at first sign",
        "Avoid overhead watering"
    ],
    "Strawberry Leaf Scorch": [
        "Remove infected leaves",
        "Maintain proper plant spacing",
        "Ensure good air circulation",
        "Apply appropriate fungicides"
    ],
    "Strawberry Healthy": [
        "Regular renovation",
        "Proper irrigation practices",
        "Maintain soil fertility",
        "Monitor for pest issues"
    ],
    "Tomato Bacterial Spot": [
        "Use disease-free seeds",
        "Rotate crops",
        "Avoid overhead irrigation",
        "Apply copper-based sprays"
    ],
    "Tomato Early Blight": [
        "Remove lower infected leaves",
        "Mulch around plants",
        "Apply appropriate fungicides",
        "Maintain plant spacing"
    ],
    "Tomato Late Blight": [
        "Monitor weather conditions",
        "Apply preventive fungicides",
        "Remove infected plants",
        "Improve air circulation"
    ],
    "Tomato Leaf Mold": [
        "Reduce humidity levels",
        "Improve air circulation",
        "Remove infected leaves",
        "Apply appropriate fungicides"
    ],
    "Tomato Septoria Leaf Spot": [
        "Remove infected leaves",
        "Mulch around plants",
        "Apply fungicides preventively",
        "Maintain proper spacing"
    ],
    "Tomato Spider Mites": [
        "Increase humidity levels",
        "Use insecticidal soaps",
        "Introduce predatory mites",
        "Remove heavily infested leaves"
    ],
    "Tomato Target Spot": [
        "Improve air circulation",
        "Remove infected leaves",
        "Apply appropriate fungicides",
        "Avoid overhead irrigation"
    ],
    "Tomato Yellow Leaf Curl Virus": [
        "Control whitefly populations",
        "Use virus-resistant varieties",
        "Remove infected plants",
        "Use reflective mulches"
    ],
    "Tomato Mosaic Virus": [
        "Remove infected plants immediately",
        "Control weed hosts",
        "Use virus-resistant varieties",
        "Sanitize tools between plants"
    ],
    "Tomato Healthy": [
        "Regular pruning",
        "Proper fertilization",
        "Monitor for pest issues",
        "Maintain proper irrigation"
    ]
}

@app.route('/', methods=['GET'])
def index():

    return render_template('index.html')
    # return """
    # <h1>Plant Disease Prediction API</h1>
    # <p>Upload an image to predict the plant disease.</p>
    # <form action="/predict" method="post" enctype="multipart/form-data">
    #     <input type="file" name="file" />
    #     <input type="submit" value="Predict" />
    # </form>
    # """


@app.route('/cropdisease', methods=['GET'])
def cropdisease():
    return render_template('crop-disease-prediction.html')

@app.route('/croprecommendation', methods=['GET'])
def croprecommendation():
    return render_template('crop-recommendation.html')

@app.route('/fertilizerrecommendation', methods=['GET'])
def fertilizerrecommendation():
    return render_template('fertilizer-recommendation.html')




from io import BytesIO
# First, define the precautions dictionary


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load image from file stream
        image = load_img(BytesIO(file.read()), target_size=IMG_SIZE)  # Use BytesIO for file-like object
        image_array = img_to_array(image)            # Convert to array
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = preprocess_input(image_array)  # Normalize for ResNet50

        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get class name and precautions
        class_name = CLASS_NAMES[predicted_class]
        precautions = DISEASE_PRECAUTIONS.get(class_name, [
            "Regular monitoring of plants",
            "Maintain good plant hygiene",
            "Follow proper irrigation practices",
            "Consult with local agricultural expert"
        ])

        # Create readable result text with precautions immediately after confidence
        result_text = f"The predicted crop disease is {class_name} with {confidence*100:.2f}% confidence.\n\nRecommended Precautions:"
        precautions_list = "\n".join([f"• {precaution}" for precaution in precautions])
        full_text = f"{result_text}\n{precautions_list}"
        
        # Return prediction result with both JSON and HTML content
        return jsonify({
            "predicted_class": class_name,
            "confidence": float(confidence),
            "precautions": f"{precautions}",
            "result_html": f"""
                <div class="prediction-result">
                    <p id="prediction-text">{result_text}</p>
                    <ul style="margin-top: 10px; margin-bottom: 15px;">
                        {' '.join([f'<li style="margin-bottom: 5px;">{precaution}</li>' for precaution in precautions])}
                    </ul>
                    <button onclick="readAloud('prediction-text')" class="read-aloud-btn">
                        <i class="fas fa-volume-up"></i> Read Results
                    </button>
                    <button onclick="stopReading()" class="stop-reading-btn">
                        <i class="fas fa-stop"></i> Stop Reading
                    </button>
                </div>
            """
        })

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

# Load the trained model
crop_model = joblib.load("models/gaussiannb_crop_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    if request.method == 'POST':
        # Get form data
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        emoji_mapping = {
            'N': ('🌿', 'N (Nitrogen ratio)'),
            'P': ('⚡', 'P (Phosphorous ratio)'),
            'K': ('💧', 'K (Potassium ratio)'),
            'temperature': ('🌡️', '°C (Temperature in Celsius)'),
            'humidity': ('💧', '% (Humidity)'),
            'ph': ('🔬', 'pH (Soil pH value)'),
            'rainfall': ('🌧️', 'mm (Rainfall in mm)')
        }


        crop_emoji_mapping = {
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

        # Prepare the input for the model
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        prediction = crop_model.predict(input_data)
        crop_name = label_encoder.inverse_transform(prediction)[0]

        return render_template('crop-recommendation.html', prediction_text=f"The most suitable crop for agriculture in the given conditions is: {crop_emoji_mapping[crop_name]} **{crop_name}** {crop_emoji_mapping[crop_name]}.")

# Load the model and scaler
fertilizer_model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Fertilizer names (decoded from label encoder)
fertilizer_names = {
    0: "Urea",
    1: "DAP",
    2: "MOP",
    3: "Complex",
    4: "SSP",
    5: "Ammonium Sulphate",
    6: "Gypsum"
}


@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    if request.method == 'POST':
        try:
            # Collect input values
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            moisture = float(request.form['moisture'])
            soil_type = int(request.form['soil_type'])
            crop_type = int(request.form['crop_type'])
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['potassium'])

            # Prepare input for prediction
            input_features = np.array([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, phosphorus, potassium]])
            input_scaled = scaler.transform(input_features)

            # Predict fertilizer
            prediction = fertilizer_model.predict(input_scaled)[0]
            fertilizer = fertilizer_names.get(prediction, "Unknown")

            return render_template('fertilizer-recommendation.html', result=f'Recommended Fertilizer: {fertilizer}')
        except Exception as e:
            return render_template('fertilizer-recommendation.html', result=f'Error: {str(e)}')
    return render_template('fertilizer-recommendation.html', result=None)


translator = Translator()

# Dictionary to map language codes
LANGUAGE_CODES = {
    'hindi': 'hi',
    'urdu': 'ur',
    'telugu': 'te'
}

def translate_text(text, target_lang):
    """
    Translate given text to target language
    """
    try:
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

@app.route('/translate', methods=['POST'])
def translate_content():
    try:
        data = request.get_json()
        text = data.get('text')
        target_lang = data.get('language')
        
        if not text or not target_lang:
            return jsonify({'error': 'Missing text or language parameter'}), 400
            
        if target_lang not in LANGUAGE_CODES:
            return jsonify({'error': 'Unsupported language'}), 400
            
        translated_text = translate_text(text, LANGUAGE_CODES[target_lang])
        return jsonify({'translated_text': translated_text})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
    