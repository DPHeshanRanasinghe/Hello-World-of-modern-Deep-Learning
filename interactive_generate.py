# Load the model and run interactive text generation

import torch
import tiktoken
from model import NanoGPT

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load checkpoint and config
checkpoint = torch.load('nanogpt_checkpoint.pt', map_location=device)
config = checkpoint['config']

# Instantiate model
model = NanoGPT(
    vocab_size=config['vocab_size'],
    n_embd=config['n_embd'],
    n_layer=config['n_layer'],
    n_head=config['n_head'],
    block_size=config['block_size'],
    dropout=config['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("Model loaded successfully!")


def generate_text(prompt, max_tokens=200, temperature=0.8, top_k=40):
    """Generate text from a given prompt."""
    model.eval()
    
    # Encode the prompt
    context = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    
    # Generate tokens
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    
    # Decode and return
    return enc.decode(generated[0].tolist())



def interactive_generate():
    """Interactive text generation interface."""
    print("Interactive Text Generation")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        prompt = input("\nEnter prompt: ")
        if prompt.lower() == 'quit':
            break
        
        try:
            temp = float(input("Temperature (0.1-2.0, default 0.8): ") or "0.8")
            tokens = int(input("Max tokens (default 200): ") or "200")
        except ValueError:
            temp, tokens = 0.8, 200
        
        print("\nGenerating...")
        generated = generate_text(prompt, max_tokens=tokens, temperature=temp)
        print("\n" + generated)

# Uncomment to run interactive mode:
interactive_generate()