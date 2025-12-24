import gradio as gr
import torch
from load_model import load_model_and_tokenizer

# --- 1. Load Model and Tokenizer ---
# This is done once when the Gradio app starts.
try:
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model, tokenizer = None, None

# --- 2. Define the Prediction Function ---
# This function is called every time a user interacts with the demo.
def predict_futures(text):
    """
    Takes raw text input, tokenizes it, gets model predictions,
    and formats the output for the Gradio interface.
    """
    if not model or not tokenizer:
        return "Model not loaded. Please check the logs.", {}

    try:
        # a. Preprocess: Tokenize the input text
        token_ids = tokenizer.encode(text)
        tokens_tensor = torch.LongTensor(token_ids).unsqueeze(0) # Add batch dimension

        # b. Predict: Get model's raw output (logits)
        with torch.no_grad():
            axis_logits, _, _ = model(tokens_tensor)
            # c. Post-process: Apply sigmoid to get probabilities (0-1)
            axis_predictions = torch.sigmoid(axis_logits)

        # d. Format Output: Create a dictionary for the label component
        axis_names = [
            "Hyper-Automation", "Human-Tech Symbiosis", "Abundance", "Individualism",
            "Community Focus", "Global Interconnectedness", "Crisis & Collapse", "Restoration & Healing",
            "Adaptation & Resilience", "Digital Dominance", "Physical Embodiment", "Collaboration"
        ]

        # Create a dictionary of {label: confidence}
        confidences = {name: float(weight) for name, weight in zip(axis_names, axis_predictions[0])}

        # You can return a simple message and the formatted labels
        return "Prediction complete.", confidences

    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"An error occurred: {e}", {}

# --- 3. Create and Launch the Gradio Interface ---
print("Creating Gradio interface...")

# Define the input and output components
input_text = gr.Textbox(
    lines=5,
    label="Input Scenario",
    placeholder="Describe a future scenario here..."
)

output_text = gr.Textbox(label="Status")
output_labels = gr.Label(label="Predicted Axis Weights", num_top_classes=12)

# Build the interface
demo = gr.Interface(
    fn=predict_futures,
    inputs=input_text,
    outputs=[output_text, output_labels],
    title="Futures Prediction Model",
    description=(
        "Explore multi-dimensional futures. "
        "Write a text describing a potential future scenario and see how the model scores it "
        "across 12 different axes, from 'Hyper-Automation' to 'Crisis & Collapse'."
    ),
    examples=[
        ["In a future dominated by hyper-automation, societal structures adapt to new forms of labor and community."],
        ["Coastal cities adopt divergent strategies as sea levels rise. Singapore invests in autonomous seawall monitoring, while Jakarta facilitates managed retreat."],
        ["A global pandemic leads to a surge in community-focused initiatives and a renewed appreciation for local supply chains."]
    ]
)

if __name__ == "__main__":
    print("Launching Gradio demo...")
    # The launch() command creates a shareable link to the demo.
    demo.launch()
