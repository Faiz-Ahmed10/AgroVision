# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# import os

# # Initialize the Flask app
# app = Flask(__name__)

# # Load the trained model
# MODEL_PATH = "RESNET50_PLANT_DISEASE.h5"
# model = load_model(MODEL_PATH)

# # Define image size and class names (update these based on your dataset)
# IMG_SIZE = (224, 224)
# CLASS_NAMES = [
#     "Class 0",  # Replace with actual class names
#     "Class 1",  # Replace with actual class names
#     # Add more classes as needed...
# ]

# @app.route('/', methods=['GET'])
# def index():
#     return "Plant Disease Prediction API is Running!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400
    
#     try:
#         # Load and preprocess the image
#         image = load_img(file, target_size=IMG_SIZE)
#         image_array = img_to_array(image) / 255.0  # Normalize the image
#         image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

#         # Make prediction
#         predictions = model.predict(image_array)
#         predicted_class = np.argmax(predictions[0])
#         confidence = predictions[0][predicted_class]

#         return jsonify({
#             "predicted_class": CLASS_NAMES[predicted_class],
#             "confidence": float(confidence)
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# from io import BytesIO

# # Initialize the Flask app
# app = Flask(__name__)

# # Load the pre-trained ResNet50 model
# MODEL_PATH = "plant_disease_resnet50.h5"
# model = load_model(MODEL_PATH)

# # Define image size and class names
# IMG_SIZE = (224, 224)  # Input size for ResNet50
# CLASS_NAMES = [
#     "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
#     "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy", 
#     "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight",
#     "Corn Healthy", "Grape Black Rot", "Grape Esca (Black Measles)",
#     "Grape Leaf Blight", "Grape Healthy", "Orange Haunglongbing (Citrus Greening)",
#     "Peach Bacterial Spot", "Peach Healthy", "Pepper Bell Bacterial Spot", 
#     "Pepper Bell Healthy", "Potato Early Blight", "Potato Late Blight", 
#     "Potato Healthy", "Raspberry Healthy", "Soybean Healthy", 
#     "Squash Powdery Mildew", "Strawberry Leaf Scorch", "Strawberry Healthy",
#     "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
#     "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
#     "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", 
#     "Tomato Mosaic Virus", "Tomato Healthy"
# ]

# @app.route('/', methods=['GET'])
# def index():
#     return """
#     <h1>Plant Disease Prediction API</h1>
#     <p>Upload an image to predict the plant disease.</p>
#     <form action="/predict" method="post" enctype="multipart/form-data">
#         <input type="file" name="file" />
#         <input type="submit" value="Predict" />
#     </form>
#     """


# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     Endpoint to classify uploaded plant leaf images.
#     """
#     # Check if the request contains a file
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     try:
#         # Load and preprocess the image using BytesIO
#         image = load_img(BytesIO(file.read()), target_size=IMG_SIZE)  # Fix here
#         image_array = img_to_array(image)  # Convert image to array
#         image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#         image_array = preprocess_input(image_array)  # Preprocess for ResNet50

#         # Make prediction
#         predictions = model.predict(image_array)
#         predicted_class_idx = np.argmax(predictions[0])  # Get class index
#         confidence = predictions[0][predicted_class_idx]  # Confidence score

#         # Return the result as JSON
#         return jsonify({
#             "predicted_class": CLASS_NAMES[predicted_class_idx],
#             "confidence": float(confidence)
#         })

#     except Exception as e:
#         # Handle errors
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     # Run the Flask server
#     app.run(debug=True)





from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the trained .keras model
MODEL_PATH = "final.keras"
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

        # Return prediction result
        return jsonify({
            "predicted_class": CLASS_NAMES[predicted_class],
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# import pickle

# MODEL_PATH = 'crop_recommendation.pkl'
# with open(MODEL_PATH, 'rb') as model_file:
#     model = pickle.load(model_file)

# # Crop dictionary
# crop_dict = {
#     'rice': 1,
#     'maize': 2,
#     'chickpea': 3,
#     'kidneybeans': 4,
#     'pigeonpeas': 5,
#     'mothbeans': 6,
#     'mungbean': 7,
#     'blackgram': 8,
#     'lentil': 9,
#     'pomegranate': 10,
#     'banana': 11,
#     'mango': 12,
#     'grapes': 13,
#     'watermelon': 14,
#     'muskmelon': 15,
#     'apple': 16,
#     'orange': 17,
#     'papaya': 18,
#     'coconut': 19,
#     'cotton': 20,
#     'jute': 21,
#     'coffee': 22
# }

# # Reverse the crop dictionary
# reverse_crop_dict = {v: k for k, v in crop_dict.items()}




# @app.route('/recommend', methods=['POST'])
# def recommend():
#     try:
#         # Parse JSON request data
#         data = request.get_json()
#         features = [
#             data['nitrogen'],
#             data['phosphorus'],
#             data['potassium'],
#             data['temperature'],
#             data['humidity'],
#             data['ph'],
#             data['rainfall']
#         ]
        
#          # Predict the recommended crop
#         predicted_value = model.predict([features])[0]
        
#         # Convert prediction to crop name
#         predicted_crop = reverse_crop_dict.get(int(predicted_value), 'Unknown Crop')

#         return jsonify({
#             'recommended_crop': predicted_crop
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)


