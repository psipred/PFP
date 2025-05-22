import torch, torch.nn as nn, torch.nn.functional as F

class BaseGOClassifier(nn.Module):
    """Simple feed-forward multi-label classifier (shared by every modality)."""
    def __init__(self, input_dim: int, output_dim: int,
                 projection_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        # Project input to common dimension
        self.input_proj = (nn.Linear(input_dim, projection_dim)
                           if input_dim != projection_dim else nn.Identity())
        # First hidden block
        self.hidden1 = nn.Linear(projection_dim, hidden_dim)
        self.bn1     = nn.BatchNorm1d(hidden_dim)
        self.dropout1= nn.Dropout(0.3)
        # Second hidden block
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2     = nn.BatchNorm1d(hidden_dim)
        self.dropout2= nn.Dropout(0.3)
        # Output layer
        self.output  = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)                         # (batch, projection_dim)
        # Block 1
        x = self.hidden1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Block 2
        x = self.hidden2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        # Output logits
        x = self.output(x)
        return x