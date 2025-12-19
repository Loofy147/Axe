"""
Build a custom vocabulary from the futures dataset.

This solves the critical issue: GPT-2's 50K vocabulary is too large
for a 256-dim model. We'll create a ~3K vocabulary from only the words
in our dataset, which the model can actually learn.
"""

import json
from collections import Counter
import pickle

def build_custom_vocabulary(dataset_path='futures_dataset.json', vocab_size=3000):
    """
    Build custom vocabulary from dataset.

    Args:
        dataset_path: Path to futures dataset JSON
        vocab_size: Target vocabulary size (not including special tokens)

    Returns:
        vocab_dict: {word: index} mapping
        idx_to_word: {index: word} mapping
    """

    print("=" * 70)
    print("Building Custom Vocabulary")
    print("=" * 70)

    # Load dataset
    print(f"\n1. Loading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        data = json.load(f)

    samples = data['samples']
    print(f"   Total samples: {len(samples)}")

    # Tokenize and count words
    print("\n2. Counting words...")
    word_counts = Counter()

    for sample in samples:
        text = sample['text'].lower()

        # Simple tokenization (split on whitespace and punctuation)
        text = text.replace(',', ' ,').replace('.', ' .')
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        text = text.replace('"', ' " ').replace("'", " ' ")

        words = text.split()
        word_counts.update(words)

    total_words = sum(word_counts.values())
    unique_words = len(word_counts)
    print(f"   Total words: {total_words:,}")
    print(f"   Unique words: {unique_words:,}")

    # Create vocabulary
    print(f"\n3. Creating vocabulary (target size: {vocab_size})...")

    # Special tokens
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']

    # Top N most common words
    top_words = [word for word, _ in word_counts.most_common(vocab_size)]

    # Combine
    vocabulary = special_tokens + top_words
    actual_vocab_size = len(vocabulary)

    # Calculate coverage
    top_word_count = sum(word_counts[w] for w in top_words)
    coverage = 100 * top_word_count / total_words

    print(f"   Final vocabulary size: {actual_vocab_size:,}")
    print(f"   Coverage: {coverage:.2f}% of dataset tokens")

    # Create mappings
    vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}
    idx_to_word = {idx: word for word, idx in vocab_dict.items()}

    # Save vocabulary
    print("\n4. Saving vocabulary...")

    with open('custom_vocab.json', 'w') as f:
        json.dump(vocab_dict, f, indent=2)

    with open('custom_vocab.pkl', 'wb') as f:
        pickle.dump({
            'vocab_dict': vocab_dict,
            'idx_to_word': idx_to_word,
            'word_counts': dict(word_counts.most_common(vocab_size)),
            'special_tokens': special_tokens
        }, f)

    print(f"   Saved to: custom_vocab.json and custom_vocab.pkl")

    # Statistics
    print("\n" + "=" * 70)
    print("Vocabulary Statistics")
    print("=" * 70)
    print(f"Vocabulary size: {actual_vocab_size:,}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Unknown token rate: {100 - coverage:.2f}%")

    # Show top 20 words
    print("\nTop 20 most common words:")
    for i, (word, count) in enumerate(word_counts.most_common(20), 1):
        print(f"  {i:2d}. {word:20s} ({count:4d} occurrences)")

    # Show some axis-specific words
    print("\nSample vocabulary by category:")
    tech_words = [w for w in top_words if any(kw in w for kw in ['ai', 'robot', 'automat', 'algorithm', 'machine', 'digital'])][:10]
    env_words = [w for w in top_words if any(kw in w for kw in ['climate', 'environment', 'carbon', 'energy', 'sustain', 'ecosystem'])][:10]
    society_words = [w for w in top_words if any(kw in w for kw in ['community', 'individual', 'global', 'collective', 'society'])][:10]

    if tech_words:
        print(f"\n  Tech words: {', '.join(tech_words)}")
    if env_words:
        print(f"  Environment words: {', '.join(env_words)}")
    if society_words:
        print(f"  Society words: {', '.join(society_words)}")

    print("\n" + "=" * 70)
    print("✓ Vocabulary creation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Use custom_vocab.pkl in training")
    print("  2. Retrain model with vocab_size = " + str(actual_vocab_size))
    print("  3. Expected: Coherent text generation + 70-80% axis accuracy")

    return vocab_dict, idx_to_word


class CustomTokenizer:
    """Simple tokenizer using custom vocabulary"""

    def __init__(self, vocab_dict):
        self.vocab_dict = vocab_dict
        self.idx_to_word = {idx: word for word, idx in vocab_dict.items()}
        self.pad_token_id = vocab_dict['<PAD>']
        self.unk_token_id = vocab_dict['<UNK>']
        self.vocab_size = len(vocab_dict)

    def encode(self, text, max_length=50):
        """Encode text to token IDs"""
        # Simple tokenization
        text = text.lower()
        text = text.replace(',', ' ,').replace('.', ' .')
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        text = text.replace('"', ' " ').replace("'", " ' ")

        words = text.split()

        # Convert to IDs
        token_ids = []
        for word in words:
            if word in self.vocab_dict:
                token_ids.append(self.vocab_dict[word])
            else:
                token_ids.append(self.unk_token_id)

        # Pad or truncate
        if len(token_ids) < max_length:
            token_ids += [self.pad_token_id] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text"""
        words = []
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']

        for idx in token_ids:
            if isinstance(idx, int):
                word = self.idx_to_word.get(idx, '<UNK>')
            else:
                word = self.idx_to_word.get(idx.item(), '<UNK>')

            if skip_special_tokens and word in special_tokens:
                continue

            words.append(word)

        # Join with spaces, but handle punctuation
        text = ' '.join(words)
        text = text.replace(' ,', ',').replace(' .', '.')
        text = text.replace('( ', '(').replace(' )', ')')

        return text
