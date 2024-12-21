from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model (ensure 'best_model.pth' is in the same directory)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load('best_model.pth', map_location=device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded image temporarily
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    # Load and preprocess the image
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # Clean up the uploaded file
    os.remove(img_path)

    # Return the predicted class index (you may want to map this to actual class names)
    return jsonify({'predicted_class': predicted.item()}), 200

if __name__ == '__main__':
    app.run(debug=True)
