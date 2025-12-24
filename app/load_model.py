import torch
from .model import FuturesModel, CustomTokenizer, build_vocabulary

def load_model_and_tokenizer(
    model_path='app/checkpoint_best.pt',
    dataset_path='app/futures_dataset_v2.json',
    vocab_size=5000,
):
    """Loads the trained FuturesModel and CustomTokenizer."""

    # 1. Build vocabulary and tokenizer
    print("Building vocabulary from dataset...")
    vocab_dict = build_vocabulary(dataset_path, vocab_size=vocab_size)
    tokenizer = CustomTokenizer(vocab_dict)
    print(f"Vocabulary size: {len(vocab_dict)}")

    # 2. Initialize the model with the same architecture
    print("Initializing model...")
    model = FuturesModel(
        vocab_size=len(vocab_dict),
        n_axes=12,
        d_model=256,
        n_head=8,
        n_layers=4,
        n_experts=8,
        dropout=0.1
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Load the saved state dictionary
    print(f"Loading model weights from {model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)

    # The state dict is nested in the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # 4. Set the model to evaluation mode
    model.eval()
    print("Model set to evaluation mode.")

    return model, tokenizer

if __name__ == "__main__":
    print("="*80)
    print("Loading Futures Prediction Model")
    print("="*80)

    # Correct paths for running from the root directory
    model_path = 'app/checkpoint_best.pt'
    dataset_path = 'app/futures_dataset_v2.json'

    try:
        model, tokenizer = load_model_and_tokenizer(
            model_path=model_path,
            dataset_path=dataset_path
        )
        print("\n✅ Model and tokenizer loaded successfully!")

        # Example usage
        print("\n--- Example Usage ---")
        text = "In a future dominated by hyper-automation, societal structures adapt to new forms of labor and community."
        print(f"Input text: '{text}'")

        token_ids = tokenizer.encode(text)
        tokens_tensor = torch.LongTensor(token_ids).unsqueeze(0) # Add batch dimension

        print(f"Encoded tokens (first 10): {tokens_tensor[0, :10]}...")

        with torch.no_grad():
            axis_logits, lm_logits, stats = model(tokens_tensor)
            axis_predictions = torch.sigmoid(axis_logits)

        print("\nPredicted Axis Weights:")
        axis_names = [
            "HyperAuto", "HumanTech", "Abundant", "Individual",
            "Community", "Global", "Crisis", "Restore",
            "Adapt", "Digital", "Physical", "Collab"
        ]
        for name, weight in zip(axis_names, axis_predictions[0]):
            print(f"  - {name:12s}: {weight:.4f}")

    except Exception as e:
        print(f"\n❌ An error occurred during loading: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
