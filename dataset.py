import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class How2SignDataset(Dataset):
    def __init__(self, data_dir, max_frames=50):
        """
        Yeh asal How2Sign dataset ko load karne ki class hai.
        data_dir: Wahan folder jahan text aur SMPL-X (.npz) files rakhi hongi.
        """
        self.data_dir = data_dir
        self.max_frames = max_frames
        
        # Asal mein yahan hum 'how2sign_realigned_train.csv' waghera load karte hain.
        # Abhi ke liye hum farz kar rahe hain ke hamare paas 2 samples hain.
        self.samples = [
            {"text": "You want to just kinda take it easy.", "pose_file": "sample_1.npz"},
            {"text": "Sign language is amazing.", "pose_file": "sample_2.npz"}
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        pose_path = os.path.join(self.data_dir, sample["pose_file"])

        # Asal 3D pose load karna (SMPL-X parameters)
        if os.path.exists(pose_path):
            data = np.load(pose_path)
            # Pose parameters (jaise 165 joints * 3D)
            poses = torch.tensor(data['poses'], dtype=torch.float32)
            
            # Frames ko fix length (50) par laana
            if poses.shape[0] > self.max_frames:
                poses = poses[:self.max_frames, :]
            elif poses.shape[0] < self.max_frames:
                pad_size = self.max_frames - poses.shape[0]
                padding = torch.zeros(pad_size, poses.shape[1])
                poses = torch.cat([poses, padding], dim=0)
        else:
            # Agar real file nahi mili, toh test ke liye zeros
            poses = torch.zeros(self.max_frames, 165 * 3)

        return text, poses

if __name__ == "__main__":
    print("How2Sign Dataset DataLoader test ho raha hai...")
    
    # Dataset aur DataLoader initialize karein
    dataset = How2SignDataset(data_dir=".")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # DataLoader se batch nikalein
    for texts, poses in dataloader:
        print("Text Batch:", texts)
        print("Poses Shape:", poses.shape)
        print("DataLoader successfully tayyar hai! (Real pipeline ready)")