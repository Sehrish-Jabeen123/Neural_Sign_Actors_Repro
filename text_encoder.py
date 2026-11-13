from transformers import CLIPModel, CLIPTokenizer
import torch

# Paper mein CLIP ViT-L/14 use hua hai
model_name = "openai/clip-vit-large-patch14"

print("CLIP Model load ho raha hai... (Thoda time lag sakta hai)")
tokenizer = CLIPTokenizer.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

def get_text_embedding(text):
    # Text ko tokens mein convert karein
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Model se sirf text features nikalein
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        
    # Naye version mein agar object return ho raha ho toh usmein se tensor nikalein
    if hasattr(text_features, 'pooler_output'):
        text_features = text_features.pooler_output
    elif hasattr(text_features, 'last_hidden_state'):
        text_features = text_features.last_hidden_state.mean(dim=1)
        
    return text_features

if __name__ == "__main__":
    print("Text Encoder (CLIP) test ho raha hai...")
    
    # Paper wala example text
    text = "You want to just kinda take it easy and then use more paint as you start filling in your details."
    
    embedding = get_text_embedding(text)
    
    print("Input Text:", text)
    print("Text Embedding Shape:", embedding.shape)
    print("Text Encoding successfully chal gaya! (CLIP implemented)")