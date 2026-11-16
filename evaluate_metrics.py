import torch
import numpy as np
import smplx
from full_model import NeuralSignActors

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralSignActors().to(device)
    # Trained model load karein
    model.load_state_dict(torch.load("sign_actors_model_3D.pth", map_location=device))
    model.eval()
    
    smplx_model = smplx.create(model_path='./models', model_type='smplx', gender='NEUTRAL', use_pca=False).to(device)
    
    text = ["This is a test sentence for evaluation."]
    
    with torch.no_grad():
        predicted_params = model(text, seq_len=1)
        
    # Asal paper ka Eq: MPVPE aur MPJPE
    # Abhi hum generated poses aur ek dummy target pose ka farq nikalenge
    target_params = torch.zeros(1, 1, 185).to(device) # Dummy target (T-pose)
    
    # Parameters se 3D mesh (vertices) aur joints nikalain
    def get_3d_output(params_tensor):
        pose = params_tensor[0, 0, :165].unsqueeze(0)    
        shape = params_tensor[0, 0, 165:175].unsqueeze(0) 
        expr = params_tensor[0, 0, 175:185].unsqueeze(0)  
        
        out = smplx_model(
            betas=shape,
            global_orient=torch.zeros(1, 3).to(device),
            body_pose=pose[:, :63],
            left_hand_pose=pose[:, 63:108],
            right_hand_pose=pose[:, 108:153],
            expression=expr
        )
        return out.vertices, out.joints

    pred_vertices, pred_joints = get_3d_output(predicted_params)
    target_vertices, target_joints = get_3d_output(target_params)
    
    # MPVPE (Mean Per Vertex Position Error) - Paper Section 5.2
    mpvpe = torch.mean(torch.norm(pred_vertices - target_vertices, dim=-1)).item()
    
    # MPJPE (Mean Per Joint Position Error)
    mpjpe = torch.mean(torch.norm(pred_joints - target_joints, dim=-1)).item()
    
    print(f"Evaluation Metrics (Paper Section 5.2):")
    print(f"MPVPE (mm): {mpvpe * 1000:.2f}") # Meters to mm
    print(f"MPJPE (mm): {mpjpe * 1000:.2f}")

if __name__ == "__main__":
    evaluate()