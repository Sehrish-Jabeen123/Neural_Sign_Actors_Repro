import pandas as pd
import os
import glob
import numpy as np
import torch
import smplx
import json

def process_video_to_3d(video_name, output_dir, max_frames=50):
    video_folder = os.path.join("openpose_output", "json", video_name)
    if not os.path.exists(video_folder):
        return None
        
    json_files = sorted(glob.glob(os.path.join(video_folder, "*.json")))[:max_frames]
    
    # SMPL-X Model Load karein
    smplx_model = smplx.create(model_path='./models', model_type='smplx', gender='NEUTRAL', use_pca=False)
    
    all_params = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if len(data['people']) == 0:
            # Agar koi insaan nahi mila toh zeros daal dein
            all_params.append(np.zeros(185))
            continue
            
        body_2d = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        
        # Optimization setup
        body_pose = torch.zeros(1, 63, requires_grad=True)
        global_orient = torch.zeros(1, 3, requires_grad=True)
        betas = torch.zeros(1, 10, requires_grad=True)
        optimizer = torch.optim.Adam([body_pose, global_orient, betas], lr=0.01)
        
        # Fitting Loop
        for i in range(50): # 50 iterations per frame (fast test)
            optimizer.zero_grad()
            output = smplx_model(betas=betas, global_orient=global_orient, body_pose=body_pose)
            joints_3d = output.joints[0, :25, :]
            joints_2d_pred = joints_3d[:, :2] * 100 + 500
            target_2d = torch.tensor(body_2d[:, :2], dtype=torch.float32)
            
            loss_rec = torch.mean(torch.abs(target_2d - joints_2d_pred))
            loss_prior = torch.mean(body_pose ** 2) + torch.mean(betas ** 2)
            loss = loss_rec + 0.01 * loss_prior
            loss.backward()
            optimizer.step()
            
        # 185 parameters save karein (63 + 3 + 10 + baqi 109 zeros)
        frame_params = np.zeros(185)
        frame_params[:63] = body_pose.detach().numpy()
        frame_params[63:66] = global_orient.detach().numpy()
        frame_params[66:76] = betas.detach().numpy()
        all_params.append(frame_params)
        
    # Padding agar frames kam hain
    while len(all_params) < max_frames:
        all_params.append(np.zeros(185))
        
    # Save as .npy
    save_path = os.path.join(output_dir, video_name + ".npy")
    np.save(save_path, np.array(all_params))
    return save_path

if __name__ == "__main__":
    print("Mini 3D Dataset banaya ja raha hai...")
    os.makedirs("mini_3d_dataset", exist_ok=True)
    
    # CSV se pehle 10 videos uthayein
    df = pd.read_csv("how2sign_train.csv", sep='\t', on_bad_lines='skip')
    if len(df.columns) == 1:
        df = pd.read_csv("how2sign_train.csv", sep=',', on_bad_lines='skip')
        
    name_col = 'SENTENCE_NAME' if 'SENTENCE_NAME' in df.columns else 'VIDEO_ID'
    
    videos = df[name_col].head(10).tolist() # Sirf 10 videos
    for i, vid in enumerate(videos):
        vid_name = str(vid).replace('.mp4', '')
        print(f"[{i+1}/10] Processing: {vid_name}")
        process_video_to_3d(vid_name, "mini_3d_dataset")
        
    print("Mini 3D Dataset tayyar ho gaya 'mini_3d_dataset' folder mein!")