import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Client-side: Lightweight embedding layer (runs on robot)"""
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.projection = nn.Linear(num_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, num_features)
        x = self.projection(x)
        x = self.norm(x)
        return x


class GeneralModel(nn.Module):
    """Server-side: Large-scale feature extractor (runs on cloud)"""
    def __init__(self, embed_dim, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Large-scale LSTM for general feature extraction
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Project bidirectional output
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * 2)
        
        out = self.projection(lstm_out)
        out = self.norm(out)
        return out


class DeformableConv1d(nn.Module):
    """Simplified deformable convolution for 1D sequences"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Learn offsets
        self.offset_conv = nn.Conv1d(in_channels, kernel_size, 1)
        # Regular convolution
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding)
        
    def forward(self, x):
        # x: (batch, channels, length)
        offsets = torch.tanh(self.offset_conv(x))  # Bounded offsets
        
        # Apply regular conv (simplified - full deformable conv is more complex)
        out = self.conv(x)
        return out


class TaskAdaptation(nn.Module):
    """Client-side: Task-specific adaptation module"""
    def __init__(self, hidden_size, num_heads=4, dropout=0.3):
        super().__init__()
        
        # Channel reduction
        self.channel_reduce = nn.Linear(hidden_size, hidden_size // 2)
        
        # 1x1 convolution for channel communication
        self.channel_conv = nn.Conv1d(hidden_size // 2, hidden_size // 2, kernel_size=1)
        
        # Deformable convolution for adaptive receptive field
        self.deform_conv = DeformableConv1d(hidden_size // 2, hidden_size // 2, kernel_size=3)
        
        # Self-attention for task interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_size // 2)
        self.norm2 = nn.LayerNorm(hidden_size // 2)
        
        # Project back to hidden_size
        self.project_out = nn.Linear(hidden_size // 2, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        batch_size, seq_len, _ = x.shape
        
        # Channel reduction
        x = self.channel_reduce(x)  # (batch, seq_len, hidden_size//2)
        
        # Channel-wise communication
        x_t = x.transpose(1, 2)  # (batch, hidden_size//2, seq_len)
        x_conv = F.gelu(self.channel_conv(x_t))
        
        # Deformable convolution
        x_deform = F.gelu(self.deform_conv(x_conv))
        x_deform = x_deform.transpose(1, 2)  # (batch, seq_len, hidden_size//2)
        
        # Residual connection
        x = self.norm1(x + x_deform)
        
        # Self-attention for task interaction
        attn_out, _ = self.attention(x, x, x)
        x = self.norm2(x + self.dropout(attn_out))
        
        # Project back
        out = self.project_out(x)
        
        return out


class TaskHead(nn.Module):
    """Client-side: Task-specific output head"""
    def __init__(self, hidden_size, num_classes, task_type='classification'):
        super().__init__()
        self.task_type = task_type
        
        if task_type == 'classification':
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, num_classes)
            )
        elif task_type == 'regression':
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1)
            )
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Use last timestep for classification
        if self.task_type == 'classification':
            x = x[:, -1, :]  # (batch, hidden_size)
        else:
            x = x.mean(dim=1)  # Global average pooling
            
        return self.head(x)


class FedBoneClient(nn.Module):
    """Complete client model: Embedding + Task Adaptation + Task Head"""
    def __init__(self, num_features, embed_dim, hidden_size, num_classes, task_type='classification'):
        super().__init__()
        
        self.embedding = PatchEmbedding(num_features, embed_dim)
        self.task_adaptation = TaskAdaptation(hidden_size)
        self.task_head = TaskHead(hidden_size, num_classes, task_type)
        
    def forward(self, x, general_features=None):
        """
        If general_features is None: return embeddings (forward pass to server)
        If general_features provided: complete task-specific forward pass
        """
        if general_features is None:
            # Step 1: Client sends embeddings to server
            embeddings = self.embedding(x)
            return embeddings
        else:
            # Step 3: Client receives features from server and completes forward
            adapted_features = self.task_adaptation(general_features)
            output = self.task_head(adapted_features)
            return output
    
    def get_client_parameters(self):
        """Get only client-side parameters (embedding, adaptation, head)"""
        params = []
        params.extend(list(self.embedding.parameters()))
        params.extend(list(self.task_adaptation.parameters()))
        params.extend(list(self.task_head.parameters()))
        return params


class FedBoneServer(nn.Module):
    """Server model: Only General Model"""
    def __init__(self, embed_dim, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        
        self.general_model = GeneralModel(embed_dim, hidden_size, num_layers, dropout)
        
    def forward(self, embeddings):
        """
        Step 2: Server processes embeddings and returns general features
        """
        features = self.general_model(embeddings)
        return features
    
    def get_general_parameters(self):
        """Get only general model parameters"""
        return list(self.general_model.parameters())


# Factory functions
def create_fedbone_client(num_features, embed_dim, hidden_size, num_classes, task_type='classification'):
    """Create FedBone client model"""
    return FedBoneClient(num_features, embed_dim, hidden_size, num_classes, task_type)


def create_fedbone_server(embed_dim, hidden_size, num_layers=2, dropout=0.3):
    """Create FedBone server model"""
    return FedBoneServer(embed_dim, hidden_size, num_layers, dropout)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)