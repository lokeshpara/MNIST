document.addEventListener('DOMContentLoaded', function() {
    // Canvas setup
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = 400;  // Increased from default
    canvas.height = 400; // Increased from default
    
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const imageUpload = document.getElementById('imageUpload');
    const predictionsDiv = document.getElementById('predictions');
    const confidenceBars = document.getElementById('confidenceBars');

    // Drawing state
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let hasDrawn = false;

    // Set canvas background to white
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Set drawing style
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 25;  // Slightly reduced line width for smoother drawing
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.globalCompositeOperation = 'source-over';

    // Show initial message
    showDrawMessage();

    // Drawing functions
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        hasDrawn = true;
        hideDrawMessage();
    }

    function stopDrawing() {
        if (isDrawing) {
            isDrawing = false;
            ctx.closePath();
        }
    }

    function draw(e) {
        if (!isDrawing) return;
        
        const [currentX, currentY] = getCoordinates(e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();
        
        [lastX, lastY] = [currentX, currentY];
        hasDrawn = true;
        hideDrawMessage();
    }

    function getCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        return [
            e.clientX - rect.left,
            e.clientY - rect.top
        ];
    }

    // Function to show draw message
    function showDrawMessage() {
        // Remove existing message if any
        const existingMessage = document.getElementById('draw-message');
        if (existingMessage) {
            existingMessage.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.id = 'draw-message';
        messageDiv.className = 'draw-message';
        messageDiv.innerHTML = `
            <div class="draw-message-content">
                <div class="draw-message-icon">✏️</div>
                <div class="draw-message-text">Draw a number here</div>
                <div class="draw-message-subtext">Use mouse or touch to draw</div>
            </div>
        `;

        // Make sure canvas container exists and has relative positioning
        const canvasContainer = canvas.parentElement;
        if (!canvasContainer) {
            console.error('Canvas container not found');
            return;
        }

        canvasContainer.style.position = 'relative';
        canvasContainer.appendChild(messageDiv);
    }

    // Function to hide draw message
    function hideDrawMessage() {
        const messageDiv = document.getElementById('draw-message');
        if (messageDiv) {
            messageDiv.remove();
        }
    }

    // Event listeners for drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch support
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);

    function handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 'mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }

    // Clear canvas
    clearBtn.addEventListener('click', clearCanvas);

    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        clearPrediction();
        showMessage('Canvas cleared!', 'success');
        hasDrawn = false;
        showDrawMessage();
    }

    // Image upload
    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = new Image();
                img.onload = () => {
                    // Clear canvas
                    ctx.fillStyle = 'white';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // Draw image maintaining aspect ratio
                    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
                    const x = (canvas.width - img.width * scale) / 2;
                    const y = (canvas.height - img.height * scale) / 2;
                    
                    ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
                    hasDrawn = true;
                    hideDrawMessage();
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    // Make prediction
    predictBtn.addEventListener('click', predictDigit);

    function clearPrediction() {
        const resultDiv = document.getElementById('prediction-result');
        resultDiv.style.display = 'none';
        document.getElementById('prediction-value').textContent = '-';
        document.getElementById('confidence-fill').style.width = '0%';
        document.getElementById('top-predictions').innerHTML = '<h4>Top Predictions</h4>';
    }

    function showMessage(message, type) {
        const errorDiv = document.getElementById('error-message');
        const successDiv = document.getElementById('success-message');
        
        if (type === 'error') {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            successDiv.style.display = 'none';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        } else {
            successDiv.textContent = message;
            successDiv.style.display = 'block';
            errorDiv.style.display = 'none';
            setTimeout(() => {
                successDiv.style.display = 'none';
            }, 3000);
        }
    }

    async function predictDigit() {
        if (!hasDrawn) {
            showMessage('Please draw a number first!', 'error');
            return;
        }

        const loading = document.querySelector('.loading');
        const selectedModel = document.getElementById('model-select').value;
        loading.style.display = 'block';
        clearPrediction();

        try {
            const imageData = canvas.toDataURL('image/png');
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    model: selectedModel
                })
            });

            const data = await response.json();
            if (data.success) {
                updatePrediction(data);
                showMessage('Prediction successful!', 'success');
            } else {
                console.error('Prediction failed:', data.error);
                showMessage('Prediction failed: ' + data.error, 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            showMessage('An error occurred during prediction', 'error');
        } finally {
            loading.style.display = 'none';
        }
    }

    function updatePrediction(data) {
        const resultDiv = document.getElementById('prediction-result');
        const modelName = document.getElementById('model-select').options[
            document.getElementById('model-select').selectedIndex
        ].text;
        
        document.getElementById('model-name').textContent = modelName;
        document.getElementById('prediction-value').textContent = data.prediction.digit;
        
        const confidence = data.prediction.confidence * 100;
        document.getElementById('confidence-fill').style.width = `${confidence}%`;
        
        const topPredictions = document.getElementById('top-predictions');
        if (data.prediction.top3_predictions && Array.isArray(data.prediction.top3_predictions)) {
            topPredictions.innerHTML = '<h4>Top Predictions</h4>' + 
                data.prediction.top3_predictions.map(p => `
                    <div class="top-prediction-item">
                        <span>Digit ${p.digit}</span>
                        <span>${(p.probability * 100).toFixed(2)}%</span>
                    </div>
                `).join('');
        } else {
            topPredictions.innerHTML = '<h4>Top Predictions</h4><p>No additional predictions available</p>';
        }
        
        resultDiv.style.display = 'block';
    }

    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'c' || e.key === 'C') {
            clearCanvas();
        } else if (e.key === 'p' || e.key === 'P') {
            predictDigit();
        }
    });

    // Add undo functionality
    let drawingHistory = [];
    let currentPath = [];

    function saveState() {
        drawingHistory.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
        currentPath = [];
    }

    function undo() {
        if (drawingHistory.length > 0) {
            ctx.putImageData(drawingHistory.pop(), 0, 0);
            showMessage('Undo successful!', 'success');
        } else {
            showMessage('Nothing to undo!', 'error');
        }
    }

    // Add keyboard shortcut for undo
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'z') {
            undo();
        }
    });

    // Add drawing guide
    function showDrawingGuide() {
        const guide = document.createElement('div');
        guide.className = 'drawing-guide';
        guide.innerHTML = `
            <div class="guide-content">
                <h3>Drawing Tips</h3>
                <ul>
                    <li>Draw digits in a single continuous motion</li>
                    <li>Keep digits centered in the canvas</li>
                    <li>Make sure digits are closed (for 0, 6, 8, 9)</li>
                    <li>Use consistent stroke width</li>
                </ul>
                <button onclick="this.parentElement.parentElement.remove()">Got it!</button>
            </div>
        `;
        document.body.appendChild(guide);
    }

    // Show guide on first visit
    if (!localStorage.getItem('guideShown')) {
        showDrawingGuide();
        localStorage.setItem('guideShown', 'true');
    }

    // Clear temporary files when leaving the page
    window.addEventListener('beforeunload', async () => {
        try {
            await fetch('/clear_temp', { method: 'POST' });
        } catch (error) {
            console.error('Error clearing temp files:', error);
        }
    });
}); 