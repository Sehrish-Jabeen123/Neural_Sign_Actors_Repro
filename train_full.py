import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from full_model import NeuralSignActors
from dataset import How2SignDataset

def train():
    print("Asal Training Pipeline shuru ho rahi hai...")

    # 1. Device Check (GPU use karne ke liye)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model is training on: {device}")

    # 2. Dataset aur DataLoader load karein
    dataset = How2SignDataset(data_dir=".")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 3. Model ko GPU par shift karein
    model = NeuralSignActors().to(device)

    # 4. Loss Function aur Optimizer
    criterion = nn.MSELoss() # Mean Squared Error (Paper mein bhi use hua hai)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. Training Loop (Yeh loop model ko baar baar data dega)
    epochs = 10
    for epoch in range(epochs):
        model.train() # Model ko training mode mein daalein
        total_loss = 0.0

        for batch_idx, (texts, target_poses) in enumerate(dataloader):
            target_poses = target_poses.to(device) # Target ko GPU par bhejein

            # Gradients ko reset karein
            optimizer.zero_grad()

            # Forward Pass: Text se poses generate karein
            predicted_poses = model(texts, seq_len=50)

            # Loss calculate karein (Prediction aur Target ka farq)
            loss = criterion(predicted_poses, target_poses)

            # Backward Pass: Model ki ghaltiyan theek karein
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Har epoch ke baad average loss print karein
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    print("Training mukammal! Model ne data se seekh liya hai.")
    
    # Model ko save karein taake baad mein use ho sake
    torch.save(model.state_dict(), "sign_actors_model.pth")
    print("Model weights 'sign_actors_model.pth' mein save ho gaye!")

if __name__ == "__main__":
    train()