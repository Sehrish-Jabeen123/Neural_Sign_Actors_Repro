import torch
from full_model import NeuralSignActors

def generate_sign_language():
    print("Model ko load kiya ja raha hai...")
    
    # 1. Model initialize karein
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralSignActors().to(device)
    
    # 2. Trained weights (dimaag) load karein jo humne abhi save kiye thay
    model.load_state_dict(torch.load("sign_actors_model.pth", map_location=device))
    model.eval() # Model ko test mode mein daalein
    
    print("Model successfully load ho gaya!")
    
    # 3. Ek naya text prompt dein jo model ne pehle nahi dekha
    test_text = ["Hello, how are you doing today?"]
    
    print(f"Input Text: {test_text[0]}")
    print("Generating 3D Sign Language Motion...")
    
    # 4. Text se 3D Motion generate karein
    with torch.no_grad():
        generated_motion = model(test_text, seq_len=50)
        
    print("Motion generated successfully!")
    print("Generated 3D Pose Shape:", generated_motion.shape)
    print("Aap ka AI asal mein sign language bana raha hai!")

if __name__ == "__main__":
    generate_sign_language()