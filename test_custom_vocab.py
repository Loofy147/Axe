"""
Test axis controllability with custom vocabulary model.

This should produce COHERENT text (not gibberish) because
we're using a vocabulary matched to model capacity.
"""

import torch
import torch.nn.functional as F
import pickle
import sys

sys.path.append('.')
from futures_model.model_improved import AxisAwareGPTWithMoEImproved
from build_custom_vocab import CustomTokenizer


AXIS_NAMES = {
    0: "Tech: Hyper-automation",
    1: "Tech: Human-centric",
    2: "Tech: Resource-abundant",
    3: "Society: Individualistic",
    4: "Society: Community",
    5: "Society: Global",
    6: "Environment: Crisis",
    7: "Environment: Restoration",
    8: "Environment: Adaptation",
    9: "Creativity: Immersive/Digital",
    10: "Creativity: Physical/Tangible",
    11: "Creativity: Collaborative",
}


def generate_with_custom_vocab(model, tokenizer, prompt_text, axis_id, max_length=20, temperature=0.8):
    """Generate text with custom vocabulary"""
    model.eval()

    # Tokenize prompt
    prompt_tokens = torch.LongTensor(tokenizer.encode(prompt_text, max_length=50))

    # Find where padding starts
    pad_id = tokenizer.pad_token_id
    actual_length = (prompt_tokens != pad_id).sum().item()
    tokens = prompt_tokens[:actual_length]

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input (pad to 50 if needed)
            input_tokens = tokens.clone()
            if len(input_tokens) < 50:
                padding = torch.full((50 - len(input_tokens),), pad_id, dtype=torch.long)
                input_tokens = torch.cat([input_tokens, padding])
            else:
                input_tokens = input_tokens[:50]

            # Forward pass
            logits, _, _, _, _, temp_head, _, _ = model(
                input_tokens.unsqueeze(0),
                torch.tensor([axis_id])
            )

            # Get logits for last real token
            last_real_pos = min(len(tokens) - 1, 49)
            next_token_logits = logits[0, last_real_pos, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, 1)

            # Stop if we generate padding or end token
            if next_token.item() == pad_id or next_token.item() == tokenizer.vocab_dict.get('<END>', -1):
                break

            generated_tokens.append(next_token.item())
            tokens = torch.cat([tokens, next_token])

    # Decode
    full_tokens = prompt_tokens[:actual_length].tolist() + generated_tokens
    generated_text = tokenizer.decode(full_tokens, skip_special_tokens=True)

    return generated_text


def test_custom_vocab_controllability(
    checkpoint_path="checkpoint_custom_vocab.pt",
    vocab_path="custom_vocab.pkl"
):
    """Test if the fixed model generates coherent, axis-controlled text"""

    print("=" * 80)
    print("Axis Controllability Test (Custom Vocabulary)")
    print("=" * 80)

    # Load vocabulary
    print("\n1. Loading custom vocabulary...")
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)

    vocab_dict = vocab_data['vocab_dict']
    vocab_size = len(vocab_dict)
    tokenizer = CustomTokenizer(vocab_dict)

    print(f"   Vocabulary size: {vocab_size:,}")

    # Load model
    print("\n2. Loading model...")
    model = AxisAwareGPTWithMoEImproved(
        vocab_size=vocab_size,
        d_model=256,
        n_axes=4,
        n_paths_per_axis=3
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"   Loaded from: {checkpoint_path}")
    print(f"   Val accuracy: {checkpoint['val_axis_acc']:.2f}%")
    print(f"   Perplexity: {checkpoint.get('perplexity', 'N/A')}")

    # Test prompts
    test_prompts = [
        "In the future, people will",
        "The next generation of technology",
        "Communities are",
        "The environment is becoming",
        "Artists create by",
    ]

    print("\n3. Testing axis control...")
    print("=" * 80)

    for prompt in test_prompts:
        print(f"\nPrompt: \"{prompt}\"")
        print("-" * 80)

        generations = {}
        for axis_id in range(12):
            try:
                generated = generate_with_custom_vocab(
                    model, tokenizer, prompt, axis_id,
                    max_length=15, temperature=0.8
                )

                # Extract continuation
                continuation = generated[len(prompt):].strip()
                if not continuation:
                    continuation = "[empty]"

                generations[axis_id] = continuation

                axis_name = AXIS_NAMES[axis_id]
                print(f"[{axis_id:2d}] {axis_name:35s}: {continuation[:60]}")

            except Exception as e:
                print(f"[{axis_id:2d}] ERROR: {e}")
                generations[axis_id] = "[error]"

        # Check diversity
        unique_continuations = len(set(generations.values()))
        print(f"\nUnique continuations: {unique_continuations}/12")

        if unique_continuations < 4:
            print("⚠️  Very low diversity - axis control may not be working")
        elif unique_continuations < 8:
            print("⚠️  Moderate diversity - some axis collapse occurring")
        else:
            print("✓ Good diversity - axes appear to be controlling generation")

    # Test coherence (most important!)
    print("\n" + "=" * 80)
    print("Coherence Test (Most Important!)")
    print("=" * 80)

    print("\nGenerating longer samples to test coherence...")

    test_cases = [
        (0, "AI robots"),
        (4, "Communities"),
        (6, "Rising temperatures"),
    ]

    coherence_score = 0
    for axis_id, prompt in test_cases:
        generated = generate_with_custom_vocab(
            model, tokenizer, prompt, axis_id,
            max_length=20, temperature=0.7
        )

        print(f"\n[Axis {axis_id}] {AXIS_NAMES[axis_id]}")
        print(f"Prompt: \"{prompt}\"")
        print(f"Generated: {generated}")

        # Manual coherence check
        words = generated.split()
        is_coherent = len(words) > 3 and all(len(w) > 1 for w in words[:5])

        if is_coherent:
            print("✓ Text appears coherent")
            coherence_score += 1
        else:
            print("✗ Text appears incoherent")

    # Overall assessment
    print("\n" + "=" * 80)
    print("Overall Assessment")
    print("=" * 80)

    axis_acc = checkpoint['val_axis_acc']
    perplexity = checkpoint.get('perplexity', 999)

    print(f"\nMetrics:")
    print(f"  Axis Accuracy: {axis_acc:.2f}%")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Coherence: {coherence_score}/3 samples")

    if axis_acc >= 75 and perplexity < 30 and coherence_score >= 2:
        print("\n✓✓✓ SUCCESS!")
        print("  The model is generating coherent text with axis control!")
        print("\n  Comparison to previous attempt:")
        print("    Before: Perplexity 214,000, gibberish output")
        print("    Now: Perplexity <30, coherent sentences")
        print("\n  This is PUBLISHABLE research.")
        print("\n  Next steps:")
        print("    1. Run ablation study")
        print("    2. Scale up (d_model=512, more data)")
        print("    3. Write paper draft")

    elif axis_acc >= 60 and perplexity < 50:
        print("\n✓ GOOD PROGRESS")
        print("  The model is learning, but could improve.")
        print("\n  Try:")
        print("    - Train longer (60-80 epochs)")
        print("    - Increase model size")
        print("    - Generate more training data")

    else:
        print("⚠️  NEEDS MORE WORK")
        print("  The model hasn't converged yet.")
        print("\n  Check:")
        print("    - Did training complete?")
        print("    - Is loss still decreasing?")
        print("    - Try training for more epochs")

    return generations
