import torch
import torch.nn as nn
import numpy as np

class SkeletonGraphStructure:
    BONES = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    
    @staticmethod
    def get_adjacency_matrix(num_joints=17):
        adj = torch.zeros(num_joints, num_joints)
        for i, j in SkeletonGraphStructure.BONES:
            adj[i, j] = 1
            adj[j, i] = 1
        adj = adj + torch.eye(num_joints)
        return adj


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias


class SpatialGraphEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=32):
        super().__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gc3 = GraphConvolution(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.register_buffer('adj', SkeletonGraphStructure.get_adjacency_matrix())
        
    def forward(self, x):
        batch_size, seq_len, num_joints, _ = x.shape
        outputs = []
        
        for t in range(seq_len):
            frame = x[:, t]
            h = self.activation(self.gc1(frame, self.adj))
            h = self.dropout(h)
            h = self.activation(self.gc2(h, self.adj))
            h = self.dropout(h)
            h = self.gc3(h, self.adj)
            outputs.append(h)
        
        return torch.stack(outputs, dim=1)


class PoseTokenizer(nn.Module):
    def __init__(self, spatial_dim=32, num_joints=17, num_tokens=2, token_dim=144):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        
        self.tokenizer = nn.Sequential(
            nn.Linear(num_joints * spatial_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_tokens * token_dim),
        )
        
    def forward(self, spatial_features):
        batch_size, seq_len = spatial_features.shape[:2]
        spatial_flat = spatial_features.reshape(batch_size, seq_len, -1)
        tokens = self.tokenizer(spatial_flat)
        return tokens.reshape(batch_size, seq_len, self.num_tokens, self.token_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
            
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TemporalTransformer(nn.Module):
    def __init__(self, token_dim=144, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Linear(token_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, token_dim)
        
    def forward(self, tokens):
        batch_size, seq_len, num_tokens, token_dim = tokens.shape
        tokens_flat = tokens.reshape(batch_size, seq_len * num_tokens, token_dim)
        
        x = self.token_embed(tokens_flat)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)
        output = self.transformer_decoder(x, memory)
        reconstructed = self.output_proj(output)
        
        return reconstructed.reshape(batch_size, seq_len, num_tokens, token_dim)


class PoseDecoder(nn.Module):
    def __init__(self, token_dim=144, num_tokens=2, num_joints=17, spatial_dim=32):
        super().__init__()
        self.num_joints = num_joints
        self.spatial_dim = spatial_dim
        
        self.detokenizer = nn.Sequential(
            nn.Linear(num_tokens * token_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_joints * spatial_dim),
        )
        
        self.gc1 = GraphConvolution(spatial_dim, 64)
        self.gc2 = GraphConvolution(64, 64)
        self.gc3 = GraphConvolution(64, 2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self.register_buffer('adj', SkeletonGraphStructure.get_adjacency_matrix())
        
    def forward(self, tokens):
        batch_size, seq_len = tokens.shape[:2]
        tokens_flat = tokens.reshape(batch_size, seq_len, -1)
        spatial_features = self.detokenizer(tokens_flat)
        spatial_features = spatial_features.reshape(batch_size, seq_len, self.num_joints, self.spatial_dim)
        
        poses = []
        for t in range(seq_len):
            frame = spatial_features[:, t]
            h = self.activation(self.gc1(frame, self.adj))
            h = self.dropout(h)
            h = self.activation(self.gc2(h, self.adj))
            h = self.dropout(h)
            pose = self.gc3(h, self.adj)
            poses.append(pose)
        
        return torch.stack(poses, dim=1)


class ShopformerV2(nn.Module):
    def __init__(self, num_joints=17, num_tokens=2, token_dim=144, d_model=256, 
                 nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.spatial_encoder = SpatialGraphEncoder(input_dim=2, hidden_dim=64, output_dim=32)
        self.tokenizer = PoseTokenizer(spatial_dim=32, num_joints=num_joints, 
                                      num_tokens=num_tokens, token_dim=token_dim)
        self.temporal_transformer = TemporalTransformer(token_dim=token_dim, d_model=d_model,
                                                       nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.pose_decoder = PoseDecoder(token_dim=token_dim, num_tokens=num_tokens,
                                       num_joints=num_joints, spatial_dim=32)
        
    def forward(self, poses):
        spatial_features = self.spatial_encoder(poses)
        tokens = self.tokenizer(spatial_features)
        reconstructed_tokens = self.temporal_transformer(tokens)
        reconstructed_poses = self.pose_decoder(reconstructed_tokens)
        return reconstructed_poses
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'total_parameters': total_params,
            'architecture': 'Spatial GCN → Tokenizer → Transformer → Decoder',
            'num_layers': 6,
            'num_heads': 8,
        }