import os
import torch
from flask import Flask, request, render_template, jsonify
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# --- 1. Initialize Application and Model ---

# Initialize the Flask app.
app = Flask(__name__)

# Define the path to the trained model.
MODEL_PATH = "potato_disease_model.pth"

# Check if the model file exists.
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please run train.py first.")

# Set the device (GPU or CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture (ViT) with the correct number of labels.
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=3)
# Load the trained weights into the model.
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# Move the model to the selected device.
model.to(device)
# Set the model to evaluation mode (important for inference).
model.eval()

# Load the image processor to preprocess images for the model.
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Define the class names in the same order as during training.
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


# --- 2. Define Flask Routes ---

@app.route('/', methods=['GET'])
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image prediction request."""
    # Check if a file was sent in the request.
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    # Check if a file was actually selected.
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        try:
            # Open the image file from the request.
            image = Image.open(file.stream).convert("RGB")

            # Preprocess the image using the ViT processor.
            inputs = processor(images=image, return_tensors="pt").to(device)
            pixel_values = inputs['pixel_values']

            # Perform inference.
            with torch.no_grad(): # Disable gradient calculation.
                outputs = model(pixel_values)
                logits = outputs.logits
            
                # --- NEW: Calculate Probabilities ---
                # Apply softmax to the logits to get probabilities.
                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
                
                # Create a dictionary of class names and their probabilities.
                all_probs = {
                    class_names[i].replace("___", " ").replace("_", " "): probabilities[i].item()
                    for i in range(len(class_names))
                }
                # --- END NEW ---

            # Return the probabilities as a JSON response.
            return jsonify({'all_probs': all_probs})

        except Exception as e:
            # Handle potential errors during prediction.
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Error processing the image.'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500


# --- 3. Run the Application ---

if __name__ == '__main__':
    # Start the Flask app. `debug=True` allows for automatic reloading on code changes.
    app.run(debug=True, host='0.0.0.0', port=5000)

