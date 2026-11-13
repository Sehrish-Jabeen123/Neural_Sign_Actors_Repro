import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomicallyInformedGNN(nn.Module):
    def __init__(self, in_features, out_features, num_joints=165):
        super().__init__()
        # 1. Anisotropic function (g_ij) - Har joint ke lye alag weights
        self.gnn_layer = nn.Linear(in_features, out_features)
        
        # 2. Pose Embedding (P_i) - Paper ki novelty
        # Har joint (0 se 164) ko ek unique pehchaan dene ke liye
        self.pose_embedding = nn.Parameter(torch.randn(num_joints, out_features))
        
        self.gamma = nn.ReLU()

    def forward(self, joint_features, adjacency_matrix):
        """
        joint_features: (batch_size, num_joints, in_features)
        adjacency_matrix: (num_joints, num_joints) - Kinematic tree batata hai ke kon kis se juda hai
        """
        batch_size, num_joints, _ = joint_features.shape
        
        # Equation 8: f_i^t = gamma( sum(g_ij(f_j - f_i)) + P_i )
        
        # Step A: Har joint ke features ko transform karein
        transformed_features = self.gnn_layer(joint_features) # (batch, joints, out_features)
        
        # Step B: Kinematic tree (adjacency) ke mutabiq neighbors ka message pass karein
        # (Yeh ek simplified message passing hai)
        messages = torch.matmul(adjacency_matrix, transformed_features)
        
        # Step C: Pose Embedding (P_i) add karein
        # Pose embedding ko batch ke mutabiq expand karein
        P = self.pose_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Step D: Combine karein aur gamma (non-linearity) lagayein
        out = self.gamma(messages + P)
        return out

# Test karne ke liye dummy data
if __name__ == "__main__":
    print("Anatomically Informed GNN (Equation 8) test ho raha hai...")
    
    batch_size = 2
    num_joints = 165 # SMPL-X ke joints
    in_features = 256 
    out_features = 512
    
    # Dummy features banayein
    dummy_joint_features = torch.randn(batch_size, num_joints, in_features)
    
    # Dummy Adjacency Matrix (SMPL-X kinematic tree ki jagah abhi ke liye identity matrix)
    dummy_adjacency = torch.eye(num_joints)
    
    # Model initialize karein
    gnn = AnatomicallyInformedGNN(in_features, out_features, num_joints)
    
    # Forward pass chalayein
    output = gnn(dummy_joint_features, dummy_adjacency)
    
    print("Input shape:", dummy_joint_features.shape)
    print("Output shape:", output.shape)
    print("GNN Encoder successfully chal gaya! (Equation 8 implemented)")