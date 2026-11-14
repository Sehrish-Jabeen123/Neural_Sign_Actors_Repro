import torch
import smplx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_real_human():
    print("SMPL-X 3D Human load ho raha hai...")
    
    # Model load karein
    model = smplx.create(model_path='./models', model_type='smplx', gender='NEUTRAL', use_pca=False)
    
    # Ek T-pose (Seedhi khari hui body) ke parameters banayein
    batch_size = 1
    body_pose = torch.zeros(batch_size, 63)
    global_orient = torch.zeros(batch_size, 3)
    betas = torch.zeros(batch_size, 10)
    
    output = model(betas=betas, global_orient=global_orient, body_pose=body_pose)
    
    # 3D joints nikalain (SMPL-X mein 144 joints hote hain)
    joints = output.joints[0].detach().cpu().numpy()
    
    print("3D Human joints extracted. Plot ban raha hai...")
    
    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # X, Y, Z extract karein
    X = joints[:, 0]
    Y = joints[:, 1]
    Z = joints[:, 2]
    
    # Plot joints
    ax.scatter(X, Y, Z, c='red', marker='o', s=10)
    
    ax.set_title("SMPL-X Real 3D Human Skeleton")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig("real_human_skeleton.png")
    print("Plot 'real_human_skeleton.png' save ho gaya!")

if __name__ == "__main__":
    plot_real_human()