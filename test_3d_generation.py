import torch
import smplx
import matplotlib.pyplot as plt
from full_model import NeuralSignActors

def generate_3d():
    print("Model aur SMPL-X load ho rahe hain...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Model Load Karein
    model = NeuralSignActors().to(device)
    
    # 2. SMPL-X 3D Body Model Load Karein
    smplx_model = smplx.create(
        model_path='./models', 
        model_type='smplx', 
        gender='NEUTRAL', 
        use_pca=False
    ).to(device)
    
    text = ["Hello, how are you doing today?"]
    print(f"Text: {text[0]}")
    
    # 3. Model se 3D parameters generate karein
    with torch.no_grad():
        params = model(text, seq_len=1) # Hum sirf 1 frame generate kar rahe hain test ke liye
        
    # Output shape: [1, 1, 185]
    # Isko alag karein: 165 (pose), 10 (shape), 10 (expression)
    pose = params[0, 0, :165].unsqueeze(0)    # 165 parameters for body + hands
    shape = params[0, 0, 165:175].unsqueeze(0) # 10 parameters for shape
    expr = params[0, 0, 175:185].unsqueeze(0)  # 10 parameters for face
    
    # SMPL-X mein body pose 63 hota hai (21 joints * 3), baqi hands ke liye
    body_pose = pose[:, :63]
    left_hand_pose = pose[:, 63:108]  # 15 joints * 3
    right_hand_pose = pose[:, 108:153] # 15 joints * 3
    
    # 4. SMPL-X ko chalayein
    output = smplx_model(
        betas=shape,
        global_orient=torch.zeros(1, 3).to(device),
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        expression=expr
    )
    
    # 5. 3D joints nikalain aur plot karein
    joints = output.joints[0].detach().cpu().numpy()
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red', s=10)
    ax.set_title("AI Generated 3D SMPL-X Skeleton")
    plt.savefig("ai_generated_3d_human.png")
    print("3D Plot 'ai_generated_3d_human.png' ke naam se save ho gaya!")

if __name__ == "__main__":
    generate_3d()