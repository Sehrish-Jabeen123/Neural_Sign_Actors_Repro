import torch
import torch.nn as nn

class AutoRegressiveDecoder(nn.Module):
    def __init__(self, text_embed_dim=768, hidden_dim=256, num_layers=4, output_pose_dim=165*3):
        super().__init__()
        # Paper Section 4.2: 4-layer LSTM
        self.lstm = nn.LSTM(
            input_size=text_embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Paper Section 4.2: MLP layer to map back to pose space
        self.mlp = nn.Linear(hidden_dim, output_pose_dim)

    def forward(self, text_embedding, seq_len):
        # Text embedding ko sequence length ke mutabiq expand karein
        # (Jaise ke 50 frames ke lye text embedding ko repeat karna)
        text_expanded = text_embedding.unsqueeze(1).repeat(1, seq_len, 1)
        
        # LSTM se guzarein
        lstm_out, (hidden, cell) = self.lstm(text_expanded)
        
        # MLP se final pose (165 joints * 3D coordinates) banayein
        poses = self.mlp(lstm_out)
        return poses

# Test karne ke liye dummy data
if __name__ == "__main__":
    print("Auto-Regressive Decoder (LSTM + MLP) test ho raha hai...")
    
    batch_size = 2
    seq_len = 50 # 50 frames generate karne hain
    text_embed_dim = 768
    
    # Dummy text embedding (jo humne CLIP se nikala tha)
    dummy_text_embedding = torch.randn(batch_size, text_embed_dim)
    
    # Model initialize karein
    decoder = AutoRegressiveDecoder(text_embed_dim=text_embed_dim)
    
    # Forward pass chalayein
    generated_poses = decoder(dummy_text_embedding, seq_len)
    
    print("Input Text Shape:", dummy_text_embedding.shape)
    print("Generated Pose Shape:", generated_poses.shape)
    print("Decoder successfully chal gaya! (LSTM + MLP implemented)")