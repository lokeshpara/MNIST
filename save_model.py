import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.04)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.04)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 8, 1, bias=False)
        )
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 14, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(0.04)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(14, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.04)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.04)
        )
        
        self.GAP = nn.AvgPool2d(6)
        self.conv8 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False)
        )
        
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

# Create model instance
model = Net()

# Load the trained weights
# Note: You'll need to copy the weights from your notebook to this location
model.load_state_dict(torch.load('models/mnist_model.pth'))

# Save the model in a format suitable for inference
model.eval()
torch.save(model.state_dict(), 'models/mnist_model_inference.pth')
print("Model saved successfully!") 