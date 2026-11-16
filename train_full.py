import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from full_model import NeuralSignActors
from dataset import How2Sign3DDataset

def train():
    print("Asal 3D SMPL-X Data par Training shuru ho rahi hai...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model is training on: {device}")

    dataset = How2Sign3DDataset(csv_file="how2sign_train.csv", data_dir="full_3d_dataset")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    model = NeuralSignActors().to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 1
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (texts, target_poses) in enumerate(dataloader):
            target_poses = target_poses.to(device)

            optimizer.zero_grad()
            # Model ab 185 parameters generate karega
            predicted_poses = model(texts, seq_len=50)

            loss = criterion(predicted_poses, target_poses)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] mukammal | Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "sign_actors_model_3D.pth")
    print("Model weights 'sign_actors_model_3D.pth' mein save ho gaye!")

if __name__ == "__main__":
    train()