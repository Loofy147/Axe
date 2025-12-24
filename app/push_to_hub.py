import os
from huggingface_hub import HfApi, HfFolder

def push_to_huggingface_hub(
    repo_name,
    username,
    folder_path='app',
    token=None
):
    """
    Pushes the contents of a folder to a new Hugging Face Hub repository.

    Args:
        repo_name (str): The name of the repository to create on the Hub.
        username (str): Your Hugging Face Hub username.
        folder_path (str, optional): The local folder to upload. Defaults to 'app'.
        token (str, optional): Your Hugging Face Hub token. If not provided,
                               it will be read from the environment or a login.
    """
    if token:
        HfFolder.save_token(token)
        print("Hugging Face token saved.")

    api = HfApi()
    repo_id = f"{username}/{repo_name}"

    # 1. Create the repository on the Hub
    print(f"Creating repository: {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        print(f"Repository '{repo_id}' created or already exists.")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return

    # 2. Upload the entire folder
    print(f"Uploading contents of '{folder_path}' to '{repo_id}'...")
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print("\n✅ Successfully uploaded files to the Hugging Face Hub!")
        print(f"Model available at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\n❌ An error occurred during upload: {e}")
        print("Please ensure your token has 'write' permissions.")

if __name__ == "__main__":
    print("="*80)
    print("Pushing Model to Hugging Face Hub")
    print("="*80)

    # --- User Configuration ---
    # Replace with your details
    HF_USERNAME = "jules-agent"  # <-- IMPORTANT: SET YOUR HF USERNAME
    HF_REPO_NAME = "futures-prediction-model" # <-- Choose a name for your model repo

    # The token is read from the environment variable for security
    HF_TOKEN = os.getenv("HF_TOKEN")

    if HF_USERNAME == "your-username" or not HF_TOKEN:
        print("\n⚠️  Please configure your Hugging Face username and token in this script.")
        print("   - Set HF_USERNAME to your username.")
        print("   - Set the HF_TOKEN environment variable with your write token.")
    else:
        push_to_huggingface_hub(
            repo_name=HF_REPO_NAME,
            username=HF_USERNAME,
            token=HF_TOKEN
        )

    print("\n" + "="*80)
