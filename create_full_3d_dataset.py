import pandas as pd
import os
import glob
import numpy as np
import torch
import smplx
import json
from tqdm import tqdm

def process_video_to_3d(video_name, output_dir, smplx_model, device, max_frames=50):
    # Resume feature: Agar file pehle se bani hai toh skip karein
    save_path = os.path.join(output_dir, video_name + ".npy")
    if os.path.exists(save_path):
        return True
        
    video_folder = os.path.join("openpose_output", "json", video_name)
    if not os.path.exists(video_folder):
        return False
        
    json_files = sorted(glob.glob(os.path.join(video_folder, "*.json")))[:max_frames]
    all_params = []
    
    # Paper Eq 4: Temporal Coherence (Pichle frame ki pose ko starting point banayein)
    prev_body_pose = torch.zeros(1, 63, device=device)
    prev_global_orient = torch.zeros(1, 3, device=device)
    prev_betas = torch.zeros(1, 10, device=device)

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if len(data['people']) == 0:
            all_params.append(np.zeros(185))
            continue
            
        body_2d = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        
        # Pichli pose se initialize karein
        body_pose = prev_body_pose.clone().detach().requires_grad_(True)
        global_orient = prev_global_orient.clone().detach().requires_grad_(True)
        betas = prev_betas.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([body_pose, global_orient, betas], lr=0.01)
        
        target_2d = torch.tensor(body_2d[:, :2], dtype=torch.float32, device=device)
        
        # Sirf 10 iterations kafi hain (Warm Start ki wajah se)
        for i in range(10): 
            optimizer.zero_grad()
            output = smplx_model(betas=betas, global_orient=global_orient, body_pose=body_pose)
            joints_3d = output.joints[0, :25, :]
            joints_2d_pred = joints_3d[:, :2] * 100 + 500
            
            loss_rec = torch.mean(torch.abs(target_2d - joints_2d_pred))
            loss_prior = torch.mean(body_pose ** 2) + torch.mean(betas ** 2)
            loss = loss_rec + 0.01 * loss_prior
            loss.backward()
            optimizer.step()
            
        # Agle frame ke liye save karein
        prev_body_pose = body_pose.detach()
        prev_global_orient = global_orient.detach()
        prev_betas = betas.detach()
            
        frame_params = np.zeros(185)
        frame_params[:63] = body_pose.detach().cpu().numpy()
        frame_params[63:66] = global_orient.detach().cpu().numpy()
        frame_params[66:76] = betas.detach().cpu().numpy()
        all_params.append(frame_params)
        
    while len(all_params) < max_frames:
        all_params.append(np.zeros(185))
        
    np.save(save_path, np.array(all_params))
    return True

if __name__ == "__main__":
    print("Strictly Paper: Full 3D Dataset Generation (GPU Accelerated) Shuru...")
    output_dir = "full_3d_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU use karein
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = pd.read_csv("how2sign_train.csv", sep='\t', on_bad_lines='skip')
    if len(df.columns) == 1:
        df = pd.read_csv("how2sign_train.csv", sep=',', on_bad_lines='skip')
        
    name_col = 'SENTENCE_NAME' if 'SENTENCE_NAME' in df.columns else 'VIDEO_ID'
    videos = df[name_col].dropna().unique().tolist()
    
    print(f"Total Videos to Process: {len(videos)}")
    
    smplx_model = smplx.create(model_path='./models', model_type='smplx', gender='NEUTRAL', use_pca=False).to(device)
    
    for vid in tqdm(videos, desc="Processing Videos"):
        vid_name = str(vid).replace('.mp4', '')
        process_video_to_3d(vid_name, output_dir, smplx_model, device)
        
    print("Poora 3D Dataset tayyar ho gaya 'full_3d_dataset' folder mein!")