import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class How2SignDataset(Dataset):
    def __init__(self, csv_file, max_frames=50):
        """
        Asal How2Sign Text Data ko load karne ki class.
        """
        self.max_frames = max_frames
        print("Asal Dataset load ho raha hai...")
        
        # CSV file ko padhein (jo humne abhi load ki hai)
        self.df = pd.read_csv(csv_file, sep='\t', on_bad_lines='skip')
        if len(self.df.columns) == 1:
            self.df = pd.read_csv(csv_file, sep=',', on_bad_lines='skip')
            
        # Text wala column dhoondein
        self.text_col = 'SENTENCE' if 'SENTENCE' in self.df.columns else self.df.columns[-1]
        print(f"Total {len(self.df)} sentences load ho gaye!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Asal text uthayein
        text = self.df.iloc[idx][self.text_col]
        
        # Abhi ke liye 3D poses dummy (zeros) hain, kyunki 21GB file download ho rahi hai
        # Jab woh file aa jayegi, hum yahan asal 3D pose load karenge
        poses = torch.zeros(self.max_frames, 165 * 3)

        return str(text), poses

if __name__ == "__main__":
    print("DataLoader ko Real Text ke sath test kiya ja raha hai...")
    
    # Dataset load karein
    dataset = How2SignDataset(csv_file="how2sign_train.csv")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test batch nikalein
    for texts, poses in dataloader:
        print("\nDataLoader se nikla hua Real Text Batch:")
        for t in texts:
            print(f"- {t}")
        print("Poses Shape:", poses.shape)
        break # Sirf pehla batch dekhne ke liye