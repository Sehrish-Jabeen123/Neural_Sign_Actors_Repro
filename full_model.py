import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

class NeuralSignActors(nn.Module):
    def __init__(self, num_joints=165, text_dim=768, hidden_dim=512):
        super().__init__()
        
        # 1. Text Encoder (CLIP)
        self.text_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # CLIP ko freeze karein
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        # 2. Anatomically Informed GNN
        self.gnn_layer = nn.Linear(hidden_dim, hidden_dim)
        self.pose_embedding = nn.Parameter(torch.randn(num_joints, hidden_dim))
        self.gamma = nn.ReLU()
        
        # 3. Auto-Regressive Decoder (LSTM + MLP)
        self.lstm = nn.LSTM(
            input_size=text_dim, 
            hidden_size=hidden_dim, 
            num_layers=4, 
            batch_first=True
        )
        self.mlp = nn.Linear(hidden_dim, num_joints * 3) 

    def forward(self, text, seq_len=50):
        # Model kis device (CPU/GPU) par hai, woh check karein
        device = next(self.parameters()).device

        # --- Step A: Text ko Embeddings mein badlein ---
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Tokens ko wahi bhejein jahan model hai (CPU ya GPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.text_model.get_text_features(**inputs)
            if hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output
            elif hasattr(text_features, 'last_hidden_state'):
                text_features = text_features.last_hidden_state.mean(dim=1)
        
        # --- Step B: LSTM Decoder ---
        text_expanded = text_features.unsqueeze(1).repeat(1, seq_len, 1)
        lstm_out, _ = self.lstm(text_expanded)
        
        # --- Step C: GNN concept ---
        gnn_out = self.gamma(self.gnn_layer(lstm_out))
        
        # --- Step D: MLP se 3D pose ---
        poses = self.mlp(gnn_out)
        
        return poses