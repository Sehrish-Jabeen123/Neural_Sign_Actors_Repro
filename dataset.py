import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import json
import glob

class How2SignDataset(Dataset):
    def __init__(self, csv_file, data_dir, max_frames=50):
        self.max_frames = max_frames
        self.data_dir = data_dir
        print("Asal OpenPose Dataset load ho raha hai...")
        
        self.df = pd.read_csv(csv_file, sep='\t', on_bad_lines='skip')
        if len(self.df.columns) == 1:
            self.df = pd.read_csv(csv_file, sep=',', on_bad_lines='skip')
            
        self.text_col = 'SENTENCE' if 'SENTENCE' in self.df.columns else self.df.columns[-1]
        self.name_col = 'SENTENCE_NAME' if 'SENTENCE_NAME' in self.df.columns else 'VIDEO_ID'
        
        print(f"Total {len(self.df)} sentences load ho gaye!")

    def __len__(self):
        return len(self.df)

    def parse_json(self, json_path):
        """Yeh function OpenPose JSON se X, Y coordinates nikalta hai"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if len(data['people']) == 0:
                return torch.zeros(165 * 3)
                
            person = data['people'][0]
            
            # Body keypoints (25 joints * 3 values = 75) -> X, Y nikalain
            body = person.get('pose_keypoints_2d', [])
            # Hand keypoints (21 joints * 3 = 63) -> X, Y nikalain
            left_hand = person.get('hand_left_keypoints_2d', [])
            right_hand = person.get('hand_right_keypoints_2d', [])
            
            # Hum confidence score (har 3rd number) ko chhor kar sirf X aur Y le rahe hain
            body_xy = [val for i, val in enumerate(body) if i % 3 != 2]
            lh_xy = [val for i, val in enumerate(left_hand) if i % 3 != 2]
            rh_xy = [val for i, val in enumerate(right_hand) if i % 3 != 2]
            
            # Sab ko mila kar ek lambi list banayein
            combined = body_xy + lh_xy + rh_xy
            
            # Humein 495 values chahiye (165 joints * 3). Agar kam hain toh 0 se bhar dein.
            tensor_pose = torch.zeros(495)
            length = min(len(combined), 495)
            tensor_pose[:length] = torch.tensor(combined[:length], dtype=torch.float32)
            
            # ==========================================
            # YEH NAYI LINE ADD KI GAYI HAI (Normalization)
            # ==========================================
            tensor_pose = tensor_pose / 1920.0 
            
            return tensor_pose
        except Exception as e:
            return torch.zeros(495)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx][self.text_col])
        video_name = str(self.df.iloc[idx][self.name_col]).replace('.mp4', '')
        
        # Asal rasta: openpose_output/json/VIDEO_NAME/
        video_folder = os.path.join(self.data_dir, "json", video_name)
        
        poses = torch.zeros(self.max_frames, 495)
        
        if os.path.exists(video_folder):
            json_files = sorted(glob.glob(os.path.join(video_folder, "*.json")))
            
            # Sirf pehle 50 frames uthayein
            for i in range(min(len(json_files), self.max_frames)):
                poses[i] = self.parse_json(json_files[i])
                
        return text, poses

if __name__ == "__main__":
    dataset = How2SignDataset(csv_file="how2sign_train.csv", data_dir="openpose_output")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for texts, poses in dataloader:
        print("\nReal Text:", texts[0])
        print("Poses Shape:", poses.shape)
        print("Kya kisi frame mein asal coordinates hain?", torch.any(poses != 0))
        break