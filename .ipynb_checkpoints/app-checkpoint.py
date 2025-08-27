from flask import Flask, request, render_template
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os

app = Flask(__name__)

# Load the model
model_path = 'D:/Crop_Disease_App/Crop_Disease_App/model/Mobilenetv2_cauliflower.pth'
model = models.mobilenet_v2(pretrained=False)
num_classes = 4  # Matches 4 classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names and recommendations
class_names = ['Healthy', 'Alternaria', 'Phosphorous Deficiency', 'Phytoxicity']
recommendations = {
    'Healthy': 'Nothing to worry.',
    'Alternaria': 'Use Plant Doctor app for treatment.',
    'Phosphorous Deficiency': 'Test soil for nutrients. Apply phosphorous-rich fertilizers.',
    'Phytoxicity': 'Reduce chemical use. Flush soil with water to remove toxins.'
}

# List of crops (single crop for now)
crops = ['Cauliflower']

@app.route('/')
def index():
    return render_template('index.html', crops=crops)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or not request.files['file'].filename:
        return render_template('index.html', crops=crops, error='Please upload an image.')
    
    if 'crop' not in request.form:
        return render_template('index.html', crops=crops, error='Please select a crop.')

    # Save the uploaded image
    file = request.files['file']
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    image_path = os.path.join(upload_folder, file.filename)
    file.save(image_path)

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
        confidence = confidence.item() * 100

    # Get recommendation
    recommendation = recommendations.get(predicted_class, 'No recommendation available.')
    selected_crop = request.form['crop']

    return render_template(
        'index.html',
        crops=crops,
        prediction=f'Predicted for {selected_crop}: {predicted_class} ({confidence:.2f}%)',
        recommendation=recommendation,
        image_file=file.filename
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)