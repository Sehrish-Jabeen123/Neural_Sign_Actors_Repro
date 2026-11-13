import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from full_model import NeuralSignActors

def visualize():
    print("Model aur Weights load ho rahe hain...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralSignActors().to(device)
    model.load_state_dict(torch.load("sign_actors_model.pth", map_location=device))
    model.eval()

    text = ["I am learning AI"]
    print("Generating motion for:", text[0])
    
    with torch.no_grad():
        motion = model(text, seq_len=50) # Shape: [1, 50, 495]

    # Shape ko [50 frames, 165 joints, 3 coordinates] mein reshape karein
    motion = motion.view(50, 165, 3).cpu().numpy()

    # Ek 3D Plot banayein (Hum pehla frame dikhayenge)
    frame_idx = 0 
    joints = motion[frame_idx]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # X, Y, Z coordinates alag karein
    X = joints[:, 0]
    Y = joints[:, 1]
    Z = joints[:, 2]

    # 3D Scatter Plot banayein
    ax.scatter(X, Y, Z, c='blue', marker='o', s=20)
    
    # Labels aur title set karein
    ax.set_title(f"AI Generated 3D Pose (Frame {frame_idx+1})")
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Plot ko save karein
    plt.savefig("generated_pose_frame0.png")
    print("Plot 'generated_pose_frame0.png' ke naam se save ho gaya hai!")
    print("Aap isko VS Code ke left side mein dekh sakti hain.")

if __name__ == "__main__":
    visualize()