import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Beta schedule (Noise kaise barhega har step par)
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def forward_process(self, x_start, t):
        """
        Equation 6: Forward Diffusion Process (Pose ko dheere dheere noise mein badalna)
        """
        noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t]).view(-1, 1, 1)
        
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy, noise

# Test karne ke liye dummy code
if __name__ == "__main__":
    print("Diffusion Model (DDPM) code test ho raha hai...")
    
    # Dummy data (Jaise ke 2 frames with 165 SMPL-X joints)
    batch_size = 2
    num_joints = 165
    x_start = torch.randn(batch_size, 1, num_joints) 
    
    # Model initialize karein
    diff_model = DiffusionModel()
    
    # Random timestep chunein (0 se 999 ke darmiyan)
    t = torch.randint(0, 1000, (batch_size,))
    
    # Forward process test karein
    x_noisy, noise = diff_model.forward_process(x_start, t)
    print("Original x shape:", x_start.shape)
    print("Noisy x shape:", x_noisy.shape)
    print("Forward process successfully chal gaya! (Equation 6 implemented)")