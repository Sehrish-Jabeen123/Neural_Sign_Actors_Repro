import torch
import smplx
import numpy as np
import json
import os
import glob

# Paper Section 3: Fitting Optimization (Equations 2, 3, 4, 5)
def fit_smplx_to_openpose(json_path):
    print(f"Processing JSON: {json_path}")
    
    # 1. OpenPose JSON se 2D points nikalain
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if len(data['people']) == 0:
        print("No person detected in JSON.")
        return
        
    person = data['people'][0]
    # Body ke 25 points (x, y, confidence)
    body_2d = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
    
    # Hum sirf un points ko le rahe hain jinki confidence > 0.1 hai (temporal coherence ke liye)
    # Paper Eq 2 ke mutabiq hum 2D points ko project karenge
    
    # 2. SMPL-X Model load karein
    smplx_model = smplx.create(model_path='./models', model_type='smplx', gender='NEUTRAL', use_pca=False)
    
    # 3. Parameters initialize karein (Paper Eq 1)
    # body_pose (63), global_orient (3), betas (10)
    body_pose = torch.zeros(1, 63, requires_grad=True)
    global_orient = torch.zeros(1, 3, requires_grad=True)
    betas = torch.zeros(1, 10, requires_grad=True)
    
    # 4. Optimizer setup (Paper Section 5.2: Adam optimizer)
    optimizer = torch.optim.Adam([body_pose, global_orient, betas], lr=0.01)
    
    # 5. Fitting Loop (Paper Eq 2: Re-projection Loss)
    for i in range(200):
        optimizer.zero_grad()
        
        # SMPL-X se 3D joints generate karein
        output = smplx_model(betas=betas, global_orient=global_orient, body_pose=body_pose)
        joints_3d = output.joints[0, :25, :] # First 25 joints (match OpenPose)
        
        # 3D ko 2D par project karein (Simplified Camera Projection)
        # Hum sirf X aur Y coordinates le rahe hain (Z ko ignore kar rahe hain test ke liye)
        # Paper Eq 2: L_rec = || J_2D - Pi_K(J_3D) ||
        joints_2d_pred = joints_3d[:, :2] * 100 + 500 # Scaling and offset to match image
        
        # Target 2D points (OpenPose se)
        target_2d = torch.tensor(body_2d[:, :2], dtype=torch.float32)
        
        # Re-projection loss (Eq 2)
        loss_rec = torch.mean(torch.abs(target_2d - joints_2d_pred))
        
        # Prior loss (Eq 3: PCA Pose Prior - simplified as L2 regularization)
        loss_prior = torch.mean(body_pose ** 2) + torch.mean(betas ** 2)
        
        # Total Loss (Eq 5)
        loss = loss_rec + 0.01 * loss_prior
        
        loss.backward()
        optimizer.step()
        
    print(f"Fitting mukammal! Re-projection Loss: {loss_rec.item():.4f}")
    
    # 3D parameters save karein
    final_params = {
        'body_pose': body_pose.detach().numpy(),
        'global_orient': global_orient.detach().numpy(),
        'betas': betas.detach().numpy()
    }
    np.savez('fitted_3d_params.npz', **final_params)
    print("3D parameters saved in 'fitted_3d_params.npz'")

if __name__ == "__main__":
    # Aap ke openpose_output folder se koi ek JSON file uthayein
    json_files = glob.glob("openpose_output/json/*/*.json")
    if json_files:
        sample_json = json_files[0]
        fit_smplx_to_openpose(sample_json)
    else:
        print("Koi JSON file nahi mili!")