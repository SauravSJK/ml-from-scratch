import torch
import torch.nn as nn

def generate_causal_mask(size: int) -> torch.Tensor:
    """Generates a No-Peak mask to prevent attending to future tokens."""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

def greedy_decode(model: nn.Module, src: torch.Tensor, max_len: int, start_symbol: int, end_symbol: int, device: torch.device):
    """
    Autoregressive decoding loop.
    Feeds the model its own predictions until it outputs the EOS token.
    """
    model.eval()
    # Initialize the target sequence with the Start-of-Sequence (SOS) token
    tgt = torch.tensor([[start_symbol]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for i in range(max_len):
            mask = generate_causal_mask(tgt.size(1)).to(device)
            
            # Forward pass
            out = model(tgt, mask)
            
            # Get the most likely next token
            prob = out[:, -1, :] # Take the last predicted token step
            _, next_word = torch.max(prob, dim=1)
            
            # Append to our ongoing sequence
            tgt = torch.cat([tgt, next_word.unsqueeze(0)], dim=1)
            
            if next_word.item() == end_symbol:
                break
                
    return tgt