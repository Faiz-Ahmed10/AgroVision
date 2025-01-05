from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from googletrans import Translator
import json
import numpy as np
import joblib
from io import BytesIO

import google.generativeai as genai


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

        genai.configure(api_key="AIzaSyDVwCinSYUfR6sNb4rkA40IwGofB0fQ554")
        gen_model = genai.GenerativeModel("gemini-1.5-flash")
        class_name = CLASS_NAMES[predicted_class]
        precautions = gen_model.generate_content(f"Provide 3-4 bullet points listing only the precautions and cures for {class_name}. Do not include any additional information or explanations.")

        # Create readable result text with precautions immediately after confidence
        result_text = f"The predicted crop disease is {class_name} with {confidence*100:.2f}% confidence.\n\nRecommended Precautions:"
        # precautions_list = "\n".join([f"• {precaution}" for precaution in precautions])
        # full_text = f"{result_text}\n{precautions.text}"
        

        return jsonify({
            "predicted_class": class_name,
            "confidence": float(confidence),
            "precautions": f"""
                    <p data-translate="{precautions.text}">{precautions.text}</p>
            """,
            "result_html": f"""
                <div class="prediction-result">
                    <p id="prediction-text">{result_text}</p>
                    <ul style="margin-top: 10px; margin-bottom: 15px;">
                        {''.join([f'<li ">{precaution}</li>' for precaution in precautions])}
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


genai.configure(api_key="AIzaSyDVwCinSYUfR6sNb4rkA40IwGofB0fQ554")
gen_model2 = genai.GenerativeModel("gemini-1.5-flash")


@app.route("/chat")
def chat_page():
    # Render the chatbot.html page when accessing /chat
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    
    if not user_input:
        return jsonify({"response": "Please provide a message."})
    
    # Get a response from the Gemini API
    try:
        response = gen_model2.generate_content(user_input)
        bot_reply = response.text
    except Exception as e:
        bot_reply = f"Error: {str(e)}"
    
    return jsonify({"response": bot_reply})



if __name__ == '__main__':
    app.run(debug=True)
    