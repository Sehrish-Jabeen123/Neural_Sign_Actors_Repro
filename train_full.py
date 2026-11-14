import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from full_model import NeuralSignActors
from dataset import How2SignDataset

def train():
    print("Asal Text Data par Training shuru ho rahi hai...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model is training on: {device}")

    # 1. ASAL DATASET LOAD KAREIN (31,165 Sentences)
    dataset = How2SignDataset(csv_file="how2sign_train.csv")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True) # Batch size 8 kar diya taake GPU fast chale

    # 2. Model
    model = NeuralSignActors().to(device)

    # 3. Loss aur Optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 4. Training Loop
    epochs = 1 # Abhi sirf 1 epoch chalayenge taake test ho jaye
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (texts, target_poses) in enumerate(dataloader):
            target_poses = target_poses.to(device)

            optimizer.zero_grad()
            
            # Text se poses generate karein
            predicted_poses = model(texts, seq_len=50)

            # Loss calculate karein
            loss = criterion(predicted_poses, target_poses)

            # Backward Pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Har 500 batch ke baad status check karein
            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx+1}/{len(dataloader)} | Current Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] mukammal | Average Loss: {avg_loss:.4f}")

    print("Training mukammal! Model ne asal data se seekh liya hai.")
    
    # Model weights save karein
    torch.save(model.state_dict(), "sign_actors_model_real.pth")
    print("Model weights 'sign_actors_model_real.pth' mein save ho gaye!")

if __name__ == "__main__":
    train()