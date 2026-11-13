import torch
import smplx

def test_smplx_model():
    print("SMPL-X Model load ho raha hai...")
    
    # Hum 'models' folder ke andar 'smplx' folder se Neutral model load kar rahe hain
    model = smplx.create(
        model_path='./models', 
        model_type='smplx', 
        gender='NEUTRAL', 
        use_pca=False
    )
    
    print("Model successfully load ho gaya!")
    
    # Ek dummy 3D pose parameters banayein (Jaise paper mein hota hai)
    batch_size = 1
    # 165 joints (body, hands, face)
    body_pose = torch.zeros(batch_size, 63) # 21 joints * 3
    global_orient = torch.zeros(batch_size, 3)
    betas = torch.zeros(batch_size, 10) # Shape parameters
    
    # Model ko chalayein (In parameters se 3D mesh banega)
    output = model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose
    )
    
    # Vertices (3D points) nikalain
    vertices = output.vertices
    print("Generated 3D Body Vertices Shape:", vertices.shape)
    print("SMPL-X Integration successfully chal gaya!")

if __name__ == "__main__":
    test_smplx_model()