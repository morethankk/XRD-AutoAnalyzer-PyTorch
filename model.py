import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout layer that applies dropout during both training and inference.
    This enables uncertainty estimation through multiple forward passes.
    """
    
    def __init__(self, p=0.5):
        super(MCDropout, self).__init__()
        self.p = p
    
    def forward(self, x):
        # Always apply dropout regardless of training mode
        return F.dropout(x, p=self.p, training=True)


class XRDNet(nn.Module):
    """
    1D Convolutional Neural Network for XRD pattern classification.
    Architecture mirrors the original TensorFlow implementation.
    """
    
    def __init__(self, n_phases, dropout_rate=0.7, n_dense=[3100, 1200]):
        super(XRDNet, self).__init__()
        
        self.n_phases = n_phases
        self.dropout_rate = dropout_rate
        
        # Convolutional layers - 6 layers with decreasing kernel sizes
        # Use explicit padding to match TensorFlow 'same' padding exactly
        self.conv1 = nn.Conv1d(1, 64, kernel_size=35, stride=1, padding=17)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=30, stride=1, padding=15)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=25, stride=1, padding=12)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv1d(64, 64, kernel_size=20, stride=1, padding=10)
        self.pool4 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        
        self.conv5 = nn.Conv1d(64, 64, kernel_size=15, stride=1, padding=7)
        self.pool5 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        
        self.conv6 = nn.Conv1d(64, 64, kernel_size=10, stride=1, padding=5)
        self.pool6 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        
        # Calculate the size after convolutions and pooling
        # Starting with 1401 points, after all pooling operations
        self.flattened_size = self._get_flattened_size(1401)
        
        # Fully connected layers
        self.dropout1 = MCDropout(dropout_rate)
        self.fc1 = nn.Linear(self.flattened_size, n_dense[0])
        self.bn1 = nn.BatchNorm1d(n_dense[0])
        
        self.dropout2 = MCDropout(dropout_rate)
        self.fc2 = nn.Linear(n_dense[0], n_dense[1])
        self.bn2 = nn.BatchNorm1d(n_dense[1])
        
        self.dropout3 = MCDropout(dropout_rate)
        self.fc3 = nn.Linear(n_dense[1], n_phases)
        
    def _get_flattened_size(self, input_length=1401):
        """Calculate the size of flattened features after conv layers"""
        # Simulate forward pass to get the size
        x = torch.randn(1, 1, input_length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.pool6(F.relu(self.conv6(x)))
        return x.numel()
    
    def get_num_features(self, input_length=1401):
        """Get the number of features after flattening for a given input length"""
        return self._get_flattened_size(input_length)
    
    def forward(self, x):
        # Convolutional layers with ReLU activation and pooling
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.pool6(F.relu(self.conv6(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout and batch normalization
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        # Only apply batch normalization if we have more than one sample in the batch or we're in eval mode
        if x.size(0) > 1 or not self.training:
            x = self.bn1(x)
        
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        # Only apply batch normalization if we have more than one sample in the batch or we're in eval mode
        if x.size(0) > 1 or not self.training:
            x = self.bn2(x)
        
        x = self.dropout3(x)
        x = self.fc3(x)
        
        # Return raw logits for cross-entropy loss
        return x
