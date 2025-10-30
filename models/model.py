import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    
    def __init__(self, num_features, hidden_size, num_layers, num_classes, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #basic long short-term memory implementation
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out

def create_model(num_features, hidden_size, num_layers, num_classes, dropout=0.3):
    return LSTMModel(num_features, hidden_size, num_layers, num_classes, dropout)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)