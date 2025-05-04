from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Define the CNN model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=0, bias=False),        # conv1: 1 input, 16 output, 3x3 kernel
            nn.ReLU(),                                         # RF: 3
            nn.BatchNorm2d(16),                                # o/p: 26
            nn.Dropout2d(0.04)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=0, bias=False),       # conv2: 16 input, 32 output, 3x3 kernel
            nn.ReLU(),                                         # RF: 5
            nn.BatchNorm2d(32),                                # o/p: 24
            nn.Dropout2d(0.04)
        )
        
        # 1x1 kernel
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 8, 1, bias=False)                    # conv3: 32 input, 8 output, 1x1 kernel
        )                                                      # RF: 5, o/p: 24
        
        self.maxpool = nn.MaxPool2d(2, 2)                      # maxpooling: 2x2 kernel, stride 2
                                                              # RF: 6, o/p: 12
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 14, 3, padding=0, bias=False),        # conv4: 8 input, 14 output, 3x3 kernel
            nn.ReLU(),                                         # RF: 10
            nn.BatchNorm2d(14),                                # o/p: 10
            nn.Dropout2d(0.04)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(14, 16, 3, padding=0, bias=False),       # conv5: 14 input, 16 output, 3x3 kernel
            nn.ReLU(),                                         # RF: 14
            nn.BatchNorm2d(16),                                # o/p: 8
            nn.Dropout2d(0.04)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False)        # conv6: 16 input, 10 output, 1x1 kernel
        )                                                      # RF: 14, o/p: 8
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0, bias=False),       # conv7: 10 input, 16 output, 3x3 kernel
            nn.ReLU(),                                         # RF: 18
            nn.BatchNorm2d(16),                                # o/p: 6
            nn.Dropout2d(0.04)
        )
        
        self.GAP = nn.AvgPool2d(6)                            # Global Average Pooling
                                                              # RF: 28, o/p: 1
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False)        # conv8: 16 input, 10 output, 1x1 kernel
        )                                                      # RF: 28, o/p: 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.GAP(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

# Load all models
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    
    # Load models with different regularizations
    model_paths = {
        'no_reg': 'models/mnist_model.pth',
        'l1': 'models/mnist_model_l1.pth',
        'l2': 'models/mnist_model_l2.pth',
        'l1_l2': 'models/mnist_model_l1_l2.pth'
    }
    
    for name, path in model_paths.items():
        model = Net().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models[name] = model
    
    return models, device

# Initialize models
models, device = load_models()

# Get available models
@app.route('/get_models', methods=['GET'])
def get_models():
    model_info = {
        'no_reg': 'No Regularization',
        'l1': 'L1 Regularization',
        'l2': 'L2 Regularization',
        'l1_l2': 'L1 + L2 Regularization'
    }
    return jsonify({
        'success': True,
        'models': model_info
    })

def preprocess_image(image_data):
    try:
        # Convert base64 to image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale and resize
        image = image.convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image = np.array(image)
        
        # Invert the image (MNIST digits are white on black)
        image = 255 - image
        
        # Add padding to ensure the digit is centered
        pad_size = 4
        padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
        
        # Resize back to 28x28
        padded_image = Image.fromarray(padded_image)
        image = padded_image.resize((28, 28), Image.Resampling.LANCZOS)
        image = np.array(image)
        
        # Normalize to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Apply MNIST normalization
        image = (image - 0.1307) / 0.3081
        
        # Ensure the image has good contrast
        image = np.clip(image, -1, 1)
        
        # Convert to tensor and add batch and channel dimensions
        image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
        return image.to(device)
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise e

@app.route('/')
def home():
    # Get list of misclassified images
    misclassified_images = []
    images_dir = os.path.join('static', 'images')
    
    # Debug print
    print(f"Looking for images in: {os.path.abspath(images_dir)}")
    
    # Check if directory exists
    if os.path.exists(images_dir):
        # Get all PNG files in the directory
        for filename in os.listdir(images_dir):
            if filename.endswith('.png'):
                image_path = os.path.join('images', filename)
                print(f"Found image: {image_path}")
                misclassified_images.append(image_path)
    else:
        print(f"Directory not found: {images_dir}")
    
    print(f"Total images found: {len(misclassified_images)}")
    return render_template('index.html', misclassified_images=misclassified_images)

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/app')
def app_page():
    return render_template('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data and selected model from request
        data = request.get_json()
        image_data = data['image']
        selected_model = data.get('model', 'no_reg')  # Default to no regularization if not specified
        
        # Validate model selection
        if selected_model not in models:
            return jsonify({
                'success': False,
                'error': f'Invalid model selection. Available models: {", ".join(models.keys())}'
            })
        
        # Preprocess image
        image_tensor = preprocess_image(image_data)
        
        # Get prediction from selected model
        with torch.no_grad():
            model = models[selected_model]
            output = model(image_tensor)
            probs = torch.exp(output)  # Convert log probabilities to probabilities
            pred = output.argmax(dim=1, keepdim=True)
            confidence = probs[0][pred].item()
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probs[0], 3)
            top3_predictions = [
                {
                    'digit': int(idx.item()),  # Convert to int
                    'probability': float(prob.item())  # Convert to float
                }
                for prob, idx in zip(top3_probs, top3_indices)
            ]
            
            prediction = {
                'digit': int(pred.item()),  # Convert to int
                'confidence': float(confidence),  # Convert to float
                'probabilities': probs[0].cpu().numpy().tolist(),
                'top3_predictions': top3_predictions
            }
        
        # Save the uploaded image temporarily
        temp_dir = 'static/temp'
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, 'last_prediction.png')
        
        # Convert base64 to image and save
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        return jsonify({
            'success': True,
            'model': selected_model,
            'prediction': prediction,
            'image_path': '/static/temp/last_prediction.png'
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/clear_temp', methods=['POST'])
def clear_temp():
    try:
        temp_dir = 'static/temp'
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 