from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from googletrans import Translator
import numpy as np
import joblib
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv
import os
from config import (
    MODEL_CONFIG, CLASS_NAMES, LANGUAGE_CODES, 
    FERTILIZER_NAMES, CROP_EMOJI_MAPPING, EMOJI_MAPPING
)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize services
translator = Translator()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


# Load all models
def load_models():
    try:
        return {
            'disease_model': tf.keras.models.load_model(MODEL_CONFIG['DISEASE_MODEL_PATH']),
            'crop_model': joblib.load(MODEL_CONFIG['CROP_MODEL_PATH']),
            'label_encoder': joblib.load(MODEL_CONFIG['LABEL_ENCODER_PATH']),
            'fertilizer_model': joblib.load(MODEL_CONFIG['FERTILIZER_MODEL_PATH']),
            'scaler': joblib.load(MODEL_CONFIG['SCALER_PATH'])
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

models = load_models()

# Route handlers
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cropdisease')
def cropdisease():
    return render_template('crop-disease-prediction.html')

@app.route('/croprecommendation')
def croprecommendation():
    return render_template('crop-recommendation.html')

@app.route('/fertilizerrecommendation')
def fertilizerrecommendation():
    return render_template('fertilizer-recommendation.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Process image
        image = load_img(BytesIO(file.read()), target_size=MODEL_CONFIG['IMG_SIZE'])
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        # Make prediction
        predictions = models['disease_model'].predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get disease information
        gen_model = genai.GenerativeModel("gemini-1.5-flash")
        class_name = CLASS_NAMES[predicted_class]
        precautions = gen_model.generate_content(
            f"Provide 3-4 bullet points listing only the precautions and cures for {class_name}. "
            "Do not include any additional information or explanations."
        )

        result_text = f"The predicted crop disease is {class_name} with {confidence*100:.2f}% confidence.\n\nRecommended Precautions:"

        return jsonify({
            "predicted_class": class_name,
            "confidence": float(confidence),
            "precautions": f"<p data-translate='{precautions.text}'>{precautions.text}</p>",
            "result_html": create_result_html(result_text, precautions)
        })

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    if request.method == 'POST':
        try:
            # Get form data
            input_data = get_crop_input_data(request.form)
            
            # Make prediction
            prediction = models['crop_model'].predict(input_data)
            crop_name = models['label_encoder'].inverse_transform(prediction)[0]
            
            return render_template(
                'crop-recommendation.html', 
                prediction_text=format_crop_prediction(crop_name)
            )
        except Exception as e:
            return render_template('crop-recommendation.html', 
                                prediction_text=f"Error: {str(e)}")

@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    if request.method == 'POST':
        try:
            # Process input data
            input_features = get_fertilizer_input_data(request.form)
            input_scaled = models['scaler'].transform(input_features)
            
            # Make prediction
            prediction = models['fertilizer_model'].predict(input_scaled)[0]
            fertilizer = FERTILIZER_NAMES.get(prediction, "Unknown")
            
            return render_template('fertilizer-recommendation.html', 
                                result=f'Recommended Fertilizer: {fertilizer}')
        except Exception as e:
            return render_template('fertilizer-recommendation.html', 
                                result=f'Error: {str(e)}')
    
    return render_template('fertilizer-recommendation.html', result=None)

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


gen_model2 = genai.GenerativeModel("gemini-1.5-flash")
@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    
    if not user_input:
        return jsonify({"response": "Please provide a message."})
    
    try:
        response = gen_model2.generate_content(user_input)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

# Helper functions
def translate_text(text, target_lang):
    try:
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def create_result_html(result_text, precautions):
    return f"""
        <div class="prediction-result">
            <p id="prediction-text">{result_text}</p>
            <ul style="margin-top: 10px; margin-bottom: 15px;">
                {''.join([f'<li>{precaution}</li>' for precaution in precautions])}
            </ul>
            <button onclick="readAloud('prediction-text')" class="read-aloud-btn">
                <i class="fas fa-volume-up"></i> Read Results
            </button>
            <button onclick="stopReading()" class="stop-reading-btn">
                <i class="fas fa-stop"></i> Stop Reading
            </button>
        </div>
    """

def get_crop_input_data(form_data):
    return np.array([[
        float(form_data['nitrogen']),
        float(form_data['phosphorus']),
        float(form_data['potassium']),
        float(form_data['temperature']),
        float(form_data['humidity']),
        float(form_data['ph']),
        float(form_data['rainfall'])
    ]])

def get_fertilizer_input_data(form_data):
    return np.array([[
        float(form_data['temperature']),
        float(form_data['humidity']),
        float(form_data['moisture']),
        int(form_data['soil_type']),
        int(form_data['crop_type']),
        float(form_data['nitrogen']),
        float(form_data['phosphorus']),
        float(form_data['potassium'])
    ]])

def format_crop_prediction(crop_name):
    emoji = CROP_EMOJI_MAPPING.get(crop_name, '')
    return f"The most suitable crop for agriculture in the given conditions is: {emoji} **{crop_name}** {emoji}."

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_ENV') == 'development')