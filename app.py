import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# Set up Gemini API key
GEMINI_API_KEY = "AIzaSyAQpgc1oFLCnyAjjkjuawSB1DHlDvozYsY"
genai.configure(api_key=GEMINI_API_KEY)

# Load the disease detection model
MODEL_PATH = "models/trained_model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    print("Model summary:")
    model.summary()  # Print model architecture
except Exception as e:
    print(f"Error loading model: {e}")

# Load class names
CLASS_NAME_PATH = "models/class_name.json"
if not os.path.exists(CLASS_NAME_PATH):
    raise FileNotFoundError(f"Class names file not found at {CLASS_NAME_PATH}.")
with open(CLASS_NAME_PATH, "r") as f:
    class_name = json.load(f)
    print("Class names:", class_name)

# Function to preprocess and predict the image
def model_prediction(test_image):
    try:
        # Load and preprocess the image
        image = Image.open(test_image)
        image = image.resize((128, 128))  # Resize to match model input size
        input_arr = np.array(image) / 255.0  # Normalize
        input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch

        # Debug: Print input array shape and values
        print("Input array shape:", input_arr.shape)
        print("Input array min/max:", np.min(input_arr), np.max(input_arr))

        # Make prediction
        prediction = model.predict(input_arr)
        print("Raw prediction output:", prediction)  # Debug: Print raw prediction
        result_index = np.argmax(prediction)
        print("Predicted class index:", result_index)  # Debug: Print predicted index
        return class_name.get(str(result_index), "Unknown Disease")  # Return the disease name
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Function to get response from Gemini API with chat history
def get_gemini_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")

        # Retrieve chat history from session
        chat_history = session.get("chat_history", [])

        # Append user's message correctly formatted
        chat_history.append({"role": "user", "parts": [{"text": user_input}]})

        # Generate AI response with chat history
        response = model.generate_content(chat_history)

        # Store AI response correctly
        bot_reply = response.text if response else "Sorry, I couldn't generate a response."

        # Append bot's response to history
        chat_history.append({"role": "model", "parts": [{"text": bot_reply}]})

        # Save updated chat history in session
        session["chat_history"] = chat_history

        return bot_reply
    except Exception as e:
        return f"Error: {str(e)}"

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route("/chatbot", methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.json.get('message')
        response = get_gemini_response(user_message)
        return jsonify({'response': response})
    return render_template('chatbot.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "")
        ai_response = get_gemini_response(user_message)
        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return jsonify({"response": "Chat history cleared."})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            print("No file part in the request")  # Debugging
            return redirect(request.url)
        
        file = request.files['file']  # Access the uploaded file
        if file.filename == '':
            print("No file selected")  # Debugging
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            print(f"File received: {file.filename}")  # Debugging

            # Save the uploaded file
            uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            file_path = os.path.join(uploads_dir, file.filename)
            file.save(file_path)  # Save the file to the uploads directory
            print(f"File saved to: {file_path}")  # Debugging

            try:
                # Predict the disease
                disease_name = model_prediction(file)  # Pass the file to the prediction function
                print(f"Predicted Disease: {disease_name}")  # Debugging
            except Exception as e:
                disease_name = f"Error: {str(e)}"
                print(f"Prediction error: {e}")  # Debugging

            return render_template('disease.html', result=disease_name, image_url=file.filename)
    
    return render_template('disease.html')

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# Set up Gemini API key
GEMINI_API_KEY = "AIzaSyAQpgc1oFLCnyAjjkjuawSB1DHlDvozYsY"
genai.configure(api_key=GEMINI_API_KEY)

# Load the disease detection model
MODEL_PATH = "models/trained_model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    print("Model summary:")
    model.summary()  # Print model architecture
except Exception as e:
    print(f"Error loading model: {e}")

# Load class names
CLASS_NAME_PATH = "models/class_name.json"
if not os.path.exists(CLASS_NAME_PATH):
    raise FileNotFoundError(f"Class names file not found at {CLASS_NAME_PATH}.")
with open(CLASS_NAME_PATH, "r") as f:
    class_name = json.load(f)
    print("Class names:", class_name)

# Function to preprocess and predict the image
def model_prediction(test_image):
    try:
        # Load and preprocess the image
        image = Image.open(test_image)
        image = image.resize((128, 128))  # Resize to match model input size
        input_arr = np.array(image) / 255.0  # Normalize
        input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch

        # Debug: Print input array shape and values
        print("Input array shape:", input_arr.shape)
        print("Input array min/max:", np.min(input_arr), np.max(input_arr))

        # Make prediction
        prediction = model.predict(input_arr)
        print("Raw prediction output:", prediction)  # Debug: Print raw prediction
        result_index = np.argmax(prediction)
        print("Predicted class index:", result_index)  # Debug: Print predicted index
        return class_name.get(str(result_index), "Unknown Disease")  # Return the disease name
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Function to get response from Gemini API with chat history
def get_gemini_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")

        # Retrieve chat history from session
        chat_history = session.get("chat_history", [])

        # Append user's message correctly formatted
        chat_history.append({"role": "user", "parts": [{"text": user_input}]})

        # Generate AI response with chat history
        response = model.generate_content(chat_history)

        # Store AI response correctly
        bot_reply = response.text if response else "Sorry, I couldn't generate a response."

        # Append bot's response to history
        chat_history.append({"role": "model", "parts": [{"text": bot_reply}]})

        # Save updated chat history in session
        session["chat_history"] = chat_history

        return bot_reply
    except Exception as e:
        return f"Error: {str(e)}"

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route("/chatbot", methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.json.get('message')
        response = get_gemini_response(user_message)
        return jsonify({'response': response})
    return render_template('chatbot.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "")
        ai_response = get_gemini_response(user_message)
        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return jsonify({"response": "Chat history cleared."})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            print("No file part in the request")  # Debugging
            return redirect(request.url)
        
        file = request.files['file']  # Access the uploaded file
        if file.filename == '':
            print("No file selected")  # Debugging
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            print(f"File received: {file.filename}")  # Debugging

            # Save the uploaded file
            uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            file_path = os.path.join(uploads_dir, file.filename)
            file.save(file_path)  # Save the file to the uploads directory
            print(f"File saved to: {file_path}")  # Debugging

            try:
                # Predict the disease
                disease_name = model_prediction(file)  # Pass the file to the prediction function
                print(f"Predicted Disease: {disease_name}")  # Debugging
            except Exception as e:
                disease_name = f"Error: {str(e)}"
                print(f"Prediction error: {e}")  # Debugging

            return render_template('disease.html', result=disease_name, image_url=file.filename)
    
    return render_template('disease.html')

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)