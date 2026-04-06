import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

class SecurityLogDataset(Dataset):
    """Handles tokenization of raw text logs for HuggingFace models."""
    def __init__(self, logs: list[str], labels: list[int], tokenizer_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Pad and truncate to ensure consistent tensor sizes
        self.encodings = self.tokenizer(logs, truncation=True, padding=True, max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx: int):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def fine_tune_model(logs: list[str], labels: list[int], num_classes: int = 2):
    """
    Fine-tunes DistilBERT for Security Log Classification (e.g., Normal vs. Malicious).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Pre-trained weights and classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=num_classes
    ).to(device)
    
    dataset = SecurityLogDataset(logs, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # AdamW is standard for Transformers (includes weight decay)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    model.train()
    for epoch in range(3): # Usually 3-5 epochs is sufficient for fine-tuning
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            # Move inputs to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # HF models compute loss internally if labels are passed
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")
    
    return model