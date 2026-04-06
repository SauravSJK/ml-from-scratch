import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformer import SimpleTransformer


def generate_sorting_data(
    num_samples: int = 10000, seq_len: int = 10, vocab_size: int = 100
):
    """
    Generates synthetic data for a numeric sorting task.

    Args:
        num_samples: Number of sequences to generate.
        seq_len: Length of each numeric sequence.
        vocab_size: Range of integers (1 to vocab_size-1).

    Returns:
        X: Unsorted sequences.
        y: Sorted target sequences.
    """
    X = torch.randint(1, vocab_size, (num_samples, seq_len))
    y = torch.sort(X, dim=1)[0]
    return X, y


def main():
    # 1. Hardware Configuration
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # 2. Hyperparameters
    vocab_size = 100
    seq_len = 10
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 512
    batch_size = 64
    epochs = 50
    learning_rate = 0.0005

    print(f"Initializing Transformer training on: {device}")

    # 3. Data Ingestion
    X_raw, y_raw = generate_sorting_data(
        num_samples=10000, seq_len=seq_len, vocab_size=vocab_size
    )
    dataset = TensorDataset(X_raw, y_raw)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Model, Loss, and Optimizer
    model = SimpleTransformer(vocab_size, d_model, num_heads, num_layers, d_ff).to(
        device
    )

    # Label smoothing regularizes the model against over-confident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW is used for better weight decay handling in Transformer architectures
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 5. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Bi-directional attention (mask=None) is used to allow global context for sorting
            outputs = model(batch_X, mask=None)

            # Flatten outputs and targets for CrossEntropy calculation
            loss = criterion(outputs.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()

            # Gradient clipping to ensure training stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1:02d}/{epochs}] | Avg Loss: {total_loss / len(loader):.4f}"
            )

    # 6. Model Evaluation & Inference Demonstration
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(1, vocab_size, (1, seq_len)).to(device)
        target = torch.sort(test_input)[0]

        # Greedy selection of the highest probability tokens
        prediction_logits = model(test_input, mask=None)
        prediction = torch.argmax(prediction_logits, dim=-1)

        print("\n" + "=" * 30)
        print("MODEL INFERENCE CHECK")
        print("=" * 30)
        print(f"Input Sequence:  {test_input.cpu().numpy().flatten().tolist()}")
        print(f"Target Sequence: {target.cpu().numpy().flatten().tolist()}")
        print(f"Model Prediction: {prediction.cpu().numpy().flatten().tolist()}")
        print("=" * 30)


if __name__ == "__main__":
    main()
