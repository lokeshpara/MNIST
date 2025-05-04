document.addEventListener('DOMContentLoaded', function() {
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-pane');

    // Function to show a specific tab
    function showTab(tabId) {
        // Hide all tab contents
        tabContents.forEach(content => {
            content.classList.remove('active');
            content.style.display = 'none';
        });

        // Remove active class from all tabs
        tabs.forEach(tab => {
            tab.classList.remove('active');
        });

        // Show selected tab content
        const selectedContent = document.getElementById(tabId);
        if (selectedContent) {
            selectedContent.classList.add('active');
            selectedContent.style.display = 'block';
        }

        // Add active class to selected tab
        const selectedTab = document.querySelector(`[data-tab="${tabId}"]`);
        if (selectedTab) {
            selectedTab.classList.add('active');
        }
    }

    // Add click event listeners to tabs
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.getAttribute('data-tab');
            showTab(tabId);
        });
    });

    // Show first tab by default
    showTab('mnist-dataset');

    // Initialize step cards
    const stepCards = document.querySelectorAll('.step-card');
    stepCards.forEach(card => {
        const header = card.querySelector('.step-header');
        const content = card.querySelector('.step-content');
        const expandBtn = card.querySelector('.expand-btn');

        header.addEventListener('click', () => {
            const isActive = content.classList.contains('active');
            
            // Close all other cards
            stepCards.forEach(otherCard => {
                if (otherCard !== card) {
                    otherCard.querySelector('.step-content').classList.remove('active');
                    otherCard.querySelector('.expand-btn').classList.remove('active');
                }
            });

            // Toggle current card
            content.classList.toggle('active');
            expandBtn.classList.toggle('active');
        });
    });

    // Initialize MNIST grid visualization
    initializeMnistGrid();

    // Initialize architecture visualization
    initializeArchitectureVisualization();

    // Initialize training visualization
    initializeTrainingVisualization();

    // Initialize regularization visualization
    initializeRegularizationVisualization();

    // Initialize results visualization
    initializeResultsVisualization();

    // Initialize syntax highlighting
    document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightBlock(block);
    });
});

function initializeMnistGrid() {
    const grid = document.querySelector('.mnist-grid');
    if (!grid) return;

    // Generate random MNIST-like images
    for (let i = 0; i < 16; i++) {
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        const ctx = canvas.getContext('2d');
        
        // Generate random digit-like pattern
        for (let x = 0; x < 28; x++) {
            for (let y = 0; y < 28; y++) {
                const value = Math.random() * 255;
                ctx.fillStyle = `rgb(${value}, ${value}, ${value})`;
                ctx.fillRect(x, y, 1, 1);
            }
        }

        const img = document.createElement('img');
        img.src = canvas.toDataURL();
        grid.appendChild(img);
    }
}

function initializeArchitectureVisualization() {
    const diagram = document.querySelector('.architecture-diagram');
    if (!diagram) return;

    // Create a simple visualization of the CNN architecture
    const layers = [
        { name: 'Input', type: 'input', size: '28x28x1' },
        { name: 'Conv1', type: 'conv', size: '26x26x32' },
        { name: 'ReLU', type: 'activation' },
        { name: 'Conv2', type: 'conv', size: '24x24x64' },
        { name: 'ReLU', type: 'activation' },
        { name: 'MaxPool', type: 'pool', size: '12x12x64' },
        { name: 'Conv3', type: 'conv', size: '10x10x128' },
        { name: 'ReLU', type: 'activation' },
        { name: 'Output', type: 'output', size: '10' }
    ];

    const layerContainer = document.createElement('div');
    layerContainer.className = 'layer-visualization';

    layers.forEach(layer => {
        const layerElement = document.createElement('div');
        layerElement.className = `layer ${layer.type}`;
        layerElement.innerHTML = `
            <div class="layer-name">${layer.name}</div>
            ${layer.size ? `<div class="layer-size">${layer.size}</div>` : ''}
        `;
        layerContainer.appendChild(layerElement);
    });

    diagram.appendChild(layerContainer);
}

function initializeTrainingVisualization() {
    const chart = document.querySelector('.training-chart');
    if (!chart) return;

    // Create a chart showing training progress
    const ctx = chart.getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 20}, (_, i) => i + 1),
            datasets: [{
                label: 'Training Accuracy',
                data: Array.from({length: 20}, () => Math.random() * 0.2 + 0.8),
                borderColor: '#007bff',
                tension: 0.4
            }, {
                label: 'Validation Accuracy',
                data: Array.from({length: 20}, () => Math.random() * 0.2 + 0.8),
                borderColor: '#28a745',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

function initializeRegularizationVisualization() {
    const comparison = document.querySelector('.effect-comparison');
    if (!comparison) return;

    // Create visualizations for L1 and L2 regularization effects
    const l1Chart = document.createElement('canvas');
    const l2Chart = document.createElement('canvas');

    comparison.querySelector('.l1-effect').appendChild(l1Chart);
    comparison.querySelector('.l2-effect').appendChild(l2Chart);

    // L1 Regularization Chart
    new Chart(l1Chart.getContext('2d'), {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'L1 Regularization',
                data: Array.from({length: 50}, () => ({
                    x: Math.random() * 2 - 1,
                    y: Math.random() * 2 - 1
                })),
                backgroundColor: '#007bff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { min: -1, max: 1 },
                y: { min: -1, max: 1 }
            }
        }
    });

    // L2 Regularization Chart
    new Chart(l2Chart.getContext('2d'), {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'L2 Regularization',
                data: Array.from({length: 50}, () => ({
                    x: Math.random() * 2 - 1,
                    y: Math.random() * 2 - 1
                })),
                backgroundColor: '#28a745'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { min: -1, max: 1 },
                y: { min: -1, max: 1 }
            }
        }
    });
}

function initializeResultsVisualization() {
    const analysis = document.querySelector('.error-analysis');
    if (!analysis) return;

    // Create a confusion matrix visualization
    const matrix = document.createElement('canvas');
    analysis.appendChild(matrix);

    new Chart(matrix.getContext('2d'), {
        type: 'heatmap',
        data: {
            labels: Array.from({length: 10}, (_, i) => i.toString()),
            datasets: [{
                data: Array.from({length: 10}, () => 
                    Array.from({length: 10}, () => Math.random())
                ),
                backgroundColor: 'rgba(0, 123, 255, 0.8)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Confusion Matrix'
                }
            }
        }
    });
}

// Step content data
const stepContent = {
    step3: {
        title: "Step 1: Define Transforms",
        overview: "Data transformation is crucial for preparing the MNIST images for training.",
        what: "This step involves defining the data transformations that will be applied to the MNIST images.",
        why: "Transforms are important because:\n- They normalize the data for better training\n- Convert images to PyTorch tensors\n- Ensure consistent data format\n- Improve model convergence",
        implementation: `# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])`,
        math: "Normalization Formula:\n\nx_normalized = (x - μ) / σ\n\nWhere:\n- x: Original pixel value\n- μ: Mean (0.1307 for MNIST)\n- σ: Standard deviation (0.3081 for MNIST)",
        note: "The normalization values (0.1307, 0.3081) are the mean and standard deviation of the MNIST dataset.",
        visual: "transforms_diagram.png"
    },
    step4: {
        title: "Step 2: Load Dataset",
        overview: "Loading the MNIST dataset and creating data loaders for efficient training and testing.",
        what: "This step involves downloading the MNIST dataset and creating data loaders for batch processing.",
        why: "Proper data loading is crucial because:\n- Enables efficient batch processing\n- Reduces memory usage\n- Provides data shuffling for better training\n- Separates training and testing data",
        implementation: `# Load training data
train_dataset = datasets.MNIST(
    'data', 
    train=True, 
    download=True, 
    transform=transform
)

# Load test data
test_dataset = datasets.MNIST(
    'data', 
    train=False, 
    transform=transform
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=1000, 
    shuffle=False
)`,
        math: "Dataset Statistics:\n\n- Training set: 60,000 images\n- Test set: 10,000 images\n- Image size: 28×28 pixels\n- Number of classes: 10 (digits 0-9)\n\nBatch Processing:\n\nTotal batches = ⌈Dataset size / Batch size⌉",
        note: "The batch size of 64 is a good starting point for training, while we use a larger batch size for testing.",
        visual: "dataset_diagram.png"
    },
    step5: {
        title: "Step 3: Visualize Data",
        overview: "Visualizing the MNIST dataset helps us understand the data we're working with.",
        what: "This step involves creating visualizations of the MNIST dataset to understand its structure and content.",
        why: "Data visualization is important because:\n- Helps identify data quality issues\n- Provides insights into data distribution\n- Verifies correct data loading\n- Aids in understanding the problem",
        implementation: `def visualize_samples():
    # Get a batch of images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Create a grid of images
    fig = plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()`,
        math: "Image Processing:\n\n- Pixel values: [0, 1] after normalization\n- Grayscale: Single channel\n- Spatial dimensions: 28×28\n\nVisualization Metrics:\n\n- Grid size: 4×4\n- Figure size: 10×10 inches\n- Color map: 'gray'",
        note: "Visualization helps identify any data loading issues and provides insights into the dataset's characteristics.",
        visual: "visualization_diagram.png"
    },
    step6: {
        title: "Step 4: Define Model",
        overview: "Creating a neural network architecture suitable for MNIST digit recognition.",
        what: "This step involves defining the CNN architecture for MNIST digit recognition.",
        why: "The model architecture is crucial because:\n- Determines model capacity\n- Affects training speed\n- Influences accuracy\n- Impacts memory usage",
        implementation: `class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)`,
        math: "Convolutional Layer:\n\nOutput size = ((Input size - Kernel size + 2*Padding) / Stride) + 1\n\nMax Pooling:\n\nOutput size = Input size / Pool size\n\nReLU Activation:\n\nf(x) = max(0, x)\n\nSoftmax:\n\nσ(x)_j = e^x_j / Σ_k e^x_k",
        note: "This architecture includes convolutional layers for feature extraction and fully connected layers for classification.",
        visual: "model_architecture.png"
    },
    step7: {
        title: "Step 5: Model Summary",
        overview: "Understanding the model's architecture and its components.",
        what: "This step involves analyzing the model's structure and parameter count.",
        why: "Model analysis is important because:\n- Helps understand model complexity\n- Identifies potential bottlenecks\n- Aids in debugging\n- Guides optimization",
        implementation: `def print_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\\nModel Architecture:")
    print(model)`,
        math: "Parameter Count:\n\nTotal Parameters = Σ (Input channels × Output channels × Kernel size × Kernel size + Bias)\n\nFor each layer:\n- Conv2d: (in_channels × out_channels × kernel_size × kernel_size) + out_channels\n- Linear: (in_features × out_features) + out_features",
        note: "The model summary helps us understand the number of parameters and the flow of data through the network.",
        visual: "model_summary.png"
    },
    step8: {
        title: "Step 6: Loss & Optimizer",
        overview: "Choosing the right loss function and optimizer for training.",
        what: "This step involves selecting and configuring the loss function and optimizer.",
        why: "Loss and optimizer choice is crucial because:\n- Determines training objective\n- Affects convergence speed\n- Influences final accuracy\n- Impacts training stability",
        implementation: `# Initialize model
model = MNISTNet()

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)`,
        math: "Cross Entropy Loss:\n\nL = -Σ y_i * log(ŷ_i)\n\nWhere:\n- y_i: True label (one-hot encoded)\n- ŷ_i: Predicted probability\n\nAdam Optimizer:\n\nm_t = β₁ * m_{t-1} + (1 - β₁) * g_t\nv_t = β₂ * v_{t-1} + (1 - β₂) * g_t²\nθ_t = θ_{t-1} - α * m_t / (√v_t + ε)",
        note: "CrossEntropyLoss is suitable for multi-class classification, while Adam optimizer provides good convergence.",
        visual: "loss_optimizer.png"
    },
    step9: {
        title: "Step 7: Training Loop",
        overview: "Implementing the training loop to train the model on the MNIST dataset.",
        what: "This step involves creating the training loop that updates model parameters.",
        why: "The training loop is crucial because:\n- Updates model weights\n- Tracks training progress\n- Manages batch processing\n- Implements backpropagation",
        implementation: `def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\\tLoss: {loss.item():.6f}')`,
        math: "Backpropagation:\n\n1. Forward Pass:\n   z = Wx + b\n   a = σ(z)\n\n2. Backward Pass:\n   δ = ∂L/∂a * σ'(z)\n   ∂L/∂W = δ * x^T\n   ∂L/∂b = δ\n\n3. Weight Update:\n   W = W - α * ∂L/∂W\n   b = b - α * ∂L/∂b",
        note: "The training loop includes forward pass, loss calculation, backward pass, and parameter updates.",
        visual: "training_loop.png"
    },
    step10: {
        title: "Step 8: No Regularization",
        overview: "Training the model without any regularization to establish a baseline.",
        what: "This step involves training the model without any regularization techniques.",
        why: "Baseline training is important because:\n- Establishes performance reference\n- Helps identify overfitting\n- Guides regularization choices\n- Measures model capacity",
        implementation: `# Train without regularization
model = MNISTNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)`,
        math: "Training Metrics:\n\n1. Loss Function:\n   L = -Σ y_i * log(ŷ_i)\n\n2. Accuracy:\n   Accuracy = (Correct predictions / Total samples) × 100%\n\n3. Learning Rate:\n   α = 0.001 (Adam default)",
        note: "This baseline helps us understand the impact of regularization techniques we'll implement later.",
        visual: "no_regularization.png"
    },
    step11: {
        title: "Step 9: L1 Regularization",
        overview: "Implementing L1 regularization to prevent overfitting.",
        what: "This step involves adding L1 regularization to the training process.",
        why: "L1 regularization is important because:\n- Encourages sparsity\n- Helps feature selection\n- Reduces model complexity\n- Prevents overfitting",
        implementation: `def l1_regularization(model, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

# Add L1 regularization to training
loss = criterion(output, target) + l1_regularization(model, 0.01)`,
        math: "L1 Regularization:\n\nL1(w) = λ * Σ|w_i|\n\nWhere:\n- w_i: Model weights\n- λ: Regularization strength (0.01)\n\nTotal Loss:\n\nL_total = L_original + L1(w)",
        note: "L1 regularization helps in feature selection by encouraging sparse solutions.",
        visual: "l1_regularization.png"
    },
    step12: {
        title: "Step 10: L2 Regularization",
        overview: "Implementing L2 regularization to prevent overfitting.",
        what: "This step involves adding L2 regularization (weight decay) to the training process.",
        why: "L2 regularization is important because:\n- Prevents large weights\n- Improves generalization\n- Reduces overfitting\n- Stabilizes training",
        implementation: `# Add L2 regularization through weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)`,
        math: "L2 Regularization:\n\nL2(w) = (λ/2) * Σw_i²\n\nWhere:\n- w_i: Model weights\n- λ: Regularization strength (0.01)\n\nTotal Loss:\n\nL_total = L_original + L2(w)",
        note: "L2 regularization helps in preventing large weights and improves model stability.",
        visual: "l2_regularization.png"
    },
    step13: {
        title: "Step 11: L1 + L2 Regularization",
        overview: "Combining both L1 and L2 regularization for better performance.",
        what: "This step involves implementing both L1 and L2 regularization together.",
        why: "Combined regularization is important because:\n- Combines benefits of both methods\n- Better feature selection\n- Improved generalization\n- More robust model",
        implementation: `# Combine L1 and L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

def combined_regularization(model, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

# In training loop
loss = criterion(output, target) + combined_regularization(model, 0.01)`,
        math: "Elastic Net Regularization:\n\nL_elastic(w) = λ₁ * Σ|w_i| + (λ₂/2) * Σw_i²\n\nWhere:\n- w_i: Model weights\n- λ₁: L1 strength (0.01)\n- λ₂: L2 strength (0.01)\n\nTotal Loss:\n\nL_total = L_original + L_elastic(w)",
        note: "Combining both regularization techniques can provide better results than using either alone.",
        visual: "combined_regularization.png"
    },
    step14: {
        title: "Step 12: Evaluation",
        overview: "Evaluating the model's performance using various metrics.",
        what: "This step involves assessing the model's performance on the test set.",
        why: "Evaluation is crucial because:\n- Measures model effectiveness\n- Identifies areas for improvement\n- Validates training process\n- Guides model selection",
        implementation: `def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\\n')`,
        math: "Evaluation Metrics:\n\n1. Loss:\n   L = (1/N) * Σ L_i\n\n2. Accuracy:\n   Accuracy = (TP + TN) / (TP + TN + FP + FN)\n\n3. Error Rate:\n   Error Rate = 1 - Accuracy",
        note: "Evaluation metrics help us understand the model's performance and identify areas for improvement.",
        visual: "evaluation.png"
    },
    step15: {
        title: "Step 13: Error Analysis",
        overview: "Analyzing model errors to understand misclassifications.",
        what: "This step involves examining and understanding model errors.",
        why: "Error analysis is important because:\n- Identifies failure patterns\n- Guides model improvements\n- Reveals data issues\n- Helps debugging",
        implementation: `def analyze_errors(model, test_loader):
    model.eval()
    errors = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            mask = pred.view_as(target) != target
            errors.extend([(data[i], target[i], pred[i]) 
                          for i in range(len(mask)) if mask[i]])
    return errors`,
        math: "Error Analysis Metrics:\n\n1. Misclassification Rate:\n   MR = (FP + FN) / Total\n\n2. Per-Class Error:\n   Error_i = (FP_i + FN_i) / Total_i\n\n3. Confidence Score:\n   Confidence = max(softmax(output))",
        note: "Error analysis helps identify patterns in misclassifications and guides model improvements.",
        visual: "error_analysis.png"
    },
    step16: {
        title: "Step 14: Visualize Results",
        overview: "Creating visualizations to understand model predictions.",
        what: "This step involves creating visual representations of model results.",
        why: "Result visualization is important because:\n- Provides intuitive understanding\n- Reveals patterns\n- Aids in communication\n- Guides improvements",
        implementation: `def visualize_results(model, test_loader):
    # Get predictions
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.numpy())
            actuals.extend(target.numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.show()`,
        math: "Confusion Matrix:\n\nCM_ij = Number of samples with true label i and predicted label j\n\nMetrics:\n\n1. Precision = TP / (TP + FP)\n2. Recall = TP / (TP + FN)\n3. F1-Score = 2 * (Precision * Recall) / (Precision + Recall)",
        note: "Visualizations help in understanding model behavior and identifying areas for improvement.",
        visual: "results_visualization.png"
    },
    step17: {
        title: "Step 15: Save Model",
        overview: "Saving the trained model for future use.",
        what: "This step involves saving the model's state and parameters.",
        why: "Model saving is important because:\n- Preserves trained weights\n- Enables model reuse\n- Facilitates deployment\n- Allows transfer learning",
        implementation: `# Save the model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'mnist_model.pth')

# Load the model
checkpoint = torch.load('mnist_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']`,
        math: "Model State:\n\n1. Parameters:\n   θ = {W₁, b₁, W₂, b₂, ...}\n\n2. Optimizer State:\n   m_t, v_t (for Adam)\n\n3. Training State:\n   {epoch, loss, metrics}",
        note: "Saving the model allows us to reuse it without retraining and deploy it in production.",
        visual: "save_model.png"
    }
};

// Function to open modal with step content
function openModal(stepId) {
    const modal = document.getElementById('stepModal');
    const content = stepContent[stepId];
    
    if (content) {
        // Update modal content
        document.getElementById('modalTitle').textContent = content.title;
        document.getElementById('modalOverview').textContent = content.overview;
        document.getElementById('modalWhat').textContent = content.what;
        document.getElementById('modalWhy').textContent = content.why;
        document.getElementById('modalImplementation').textContent = content.implementation;
        if (content.math) {
            document.getElementById('modalMath').textContent = content.math;
            document.getElementById('modalMath').style.display = 'block';
        } else {
            document.getElementById('modalMath').style.display = 'none';
        }
        document.getElementById('modalNote').textContent = content.note;
        
        // Update visual content if available
        const visualElement = document.getElementById('modalVisual');
        if (content.visual) {
            visualElement.innerHTML = `<img src="/static/images/${content.visual}" alt="${content.title}">`;
        } else {
            visualElement.innerHTML = '';
        }
        
        // Show modal
        modal.style.display = 'block';
        setTimeout(() => {
            modal.classList.add('active');
        }, 10);
        
        // Highlight code
        hljs.highlightAll();
    }
}

// Function to close modal
function closeModal() {
    const modal = document.getElementById('stepModal');
    modal.classList.remove('active');
    setTimeout(() => {
        modal.style.display = 'none';
    }, 300);
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('stepModal');
    if (event.target === modal) {
        closeModal();
    }
}

// Close modal with Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeModal();
    }
});

// Initialize first step as current
document.addEventListener('DOMContentLoaded', function() {
    const firstStep = document.querySelector('.step-card');
    if (firstStep) {
        firstStep.classList.add('current');
    }
});