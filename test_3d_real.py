import torch
import smplx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from full_model import NeuralSignActors

def generate_3d_real():
    print("Trained 3D Model aur SMPL-X load ho rahe hain...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Model Load Karein
    model = NeuralSignActors().to(device)
    # Trained 3D weights load karein
    model.load_state_dict(torch.load("sign_actors_model_3D.pth", map_location=device))
    model.eval()
    
    # 2. SMPL-X Model Load Karein
    smplx_model = smplx.create(
        model_path='./models', 
        model_type='smplx', 
        gender='NEUTRAL', 
        use_pca=False
    ).to(device)
    
    text = ["Hello, how are you doing today?"]
    print(f"Text: {text[0]}")
    
    with torch.no_grad():
        # Model se 3D parameters generate karein
        params = model(text, seq_len=1) 
        
    # Output shape: [1, 1, 185]
    pose = params[0, 0, :165].unsqueeze(0)    
    shape = params[0, 0, 165:175].unsqueeze(0) 
    expr = params[0, 0, 175:185].unsqueeze(0)  
    
    body_pose = pose[:, :63]
    left_hand_pose = pose[:, 63:108] 
    right_hand_pose = pose[:, 108:153] 
    
    # 3. SMPL-X ko chalayein (Asal 3D insaan banane ke liye)
    output = smplx_model(
        betas=shape,
        global_orient=torch.zeros(1, 3).to(device),
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        expression=expr
    )
    
    joints = output.joints[0].detach().cpu().numpy()
    
    # 4. Plotting (Asal 3D Skeleton)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='blue', s=15)
    ax.set_title("AI Generated 3D SMPL-X Skeleton (Trained)")
    plt.savefig("trained_3d_human.png")
    print("3D Plot 'trained_3d_human.png' ke naam se save ho gaya!")
    print("Aap isko VS Code ke left side mein dekh sakti hain.")

if __name__ == "__main__":
    generate_3d_real()