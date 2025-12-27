import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    """
    Simple MLP probe to extract features from LLM hidden states.
    Architecture: Linear -> ReLU -> Dropout -> Linear
    """
    def __init__(self, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, hidden_state):
        # hidden_state shape: [batch, seq_len, hidden_dim]
        return self.model(hidden_state)
