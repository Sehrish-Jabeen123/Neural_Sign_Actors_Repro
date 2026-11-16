import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np

class How2Sign3DDataset(Dataset):
    def __init__(self, csv_file, data_dir, max_frames=50):
        self.max_frames = max_frames
        self.data_dir = data_dir
        print("Asal 3D SMPL-X Dataset load ho raha hai...")
        
        self.df = pd.read_csv(csv_file, sep='\t', on_bad_lines='skip')
        if len(self.df.columns) == 1:
            self.df = pd.read_csv(csv_file, sep=',', on_bad_lines='skip')
            
        self.text_col = 'SENTENCE' if 'SENTENCE' in self.df.columns else self.df.columns[-1]
        self.name_col = 'SENTENCE_NAME' if 'SENTENCE_NAME' in self.df.columns else 'VIDEO_ID'
        
        # Sirf woh videos rakhein jinka 3D data ban chuka hai
        self.valid_data = []
        for idx, row in self.df.iterrows():
            vid_name = str(row[self.name_col]).replace('.mp4', '')
            npy_path = os.path.join(self.data_dir, vid_name + ".npy")
            if os.path.exists(npy_path):
                self.valid_data.append((str(row[self.text_col]), npy_path))
                
        print(f"Total {len(self.valid_data)} videos ka asal 3D data mil gaya (baqi background mein ban raha hai)!")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        text, npy_path = self.valid_data[idx]
        # Load 3D data: Shape (50, 185)
        poses = torch.tensor(np.load(npy_path), dtype=torch.float32)
        return text, poses

if __name__ == "__main__":
    dataset = How2Sign3DDataset(csv_file="how2sign_train.csv", data_dir="full_3d_dataset")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for texts, poses in dataloader:
        print("\nReal Text:", texts[0])
        print("Real 3D Poses Shape:", poses.shape) # Yeh [4, 50, 185] aana chahiye
        break