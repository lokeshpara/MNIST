# MNIST Digit Recognition Web Application

A web application that allows users to draw handwritten digits and get predictions using a trained MNIST model.

## Features

- Interactive drawing canvas for digit input
- Real-time digit recognition
- Mobile-responsive design
- Shows top 3 predictions with confidence scores
- Support for both mouse and touch input

## Project Structure

```
mnist_app/
├── app.py              # Flask application
├── models/            # Directory for model weights
├── static/
│   ├── css/
│   │   └── style.css  # Styling
│   ├── js/
│   │   └── script.js  # Frontend logic
│   └── uploads/       # Temporary storage for uploaded images
├── templates/
│   └── index.html     # Main page template
└── requirements.txt   # Python dependencies
```

## Setup and Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the trained model weights in the `models` directory:
   ```bash
   # Copy your trained model weights to models/mnist_model.pth
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. Draw a digit (0-9) in the canvas using your mouse or touch
2. Click the "Predict" button to get predictions
3. View the top 3 predictions with their confidence scores
4. Use the "Clear" button to start over

## Model Details

The application uses a CNN model trained on the MNIST dataset with the following architecture:
- Multiple convolutional layers with batch normalization
- Dropout for regularization
- Global average pooling
- Achieves 99.36% accuracy on the test set

### Regularization Techniques Applied
The model was trained with different regularization approaches:
1. Without L1 and L2 regularization
2. With L1 regularization
3. With L2 regularization
4. With both L1 and L2 regularization

The project includes analysis of misclassified images for models with different regularization techniques.

## Technologies Used

- Backend:
  - Flask (Python web framework)
  - PyTorch (Deep learning framework)
  - NumPy (Numerical computing)

- Frontend:
  - HTML5 Canvas
  - JavaScript (ES6+)
  - CSS3

## License

This project is licensed under the MIT License - see the LICENSE file for details.
