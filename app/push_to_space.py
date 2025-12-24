import os
from huggingface_hub import HfApi, login

def deploy_to_huggingface_space(
    repo_name,
    username,
    folder_path='app',
    token=None,
    app_file="gradio_app.py"
):
    """
    Pushes the contents of a folder to a new Hugging Face Space.

    Args:
        repo_name (str): The name of the Space repository to create.
        username (str): Your Hugging Face Hub username.
        folder_path (str, optional): The local folder to upload. Defaults to 'app'.
        token (str, optional): Your Hugging Face Hub write token.
        app_file (str, optional): The main application file. Defaults to "gradio_app.py".
    """
    if not token:
        print("❌ Hugging Face token not found. Please set the HF_TOKEN environment variable.")
        return

    # 1. Log in to Hugging Face
    print("Logging in to Hugging Face Hub...")
    try:
        login(token=token, add_to_git_credential=False)
        print("✅ Login successful.")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return

    api = HfApi()
    repo_id = f"{username}/{repo_name}"

    # 2. Create the Space repository on the Hub
    print(f"Creating Space repository: {repo_id}")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )
        print(f"✅ Space repository '{repo_id}' created or already exists.")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return

    # 3. Upload the entire application folder to the Space
    print(f"Uploading contents of '{folder_path}' to '{repo_id}'...")
    try:
        # This will upload all files from the 'app' directory to the root of the Space repo
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="space",
        )
        print("\n✅ Successfully uploaded files to the Hugging Face Space!")
        print(f"Interactive demo available at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"\n❌ An error occurred during upload: {e}")
        print("Please ensure your token has 'write' permissions.")

if __name__ == "__main__":
    print("="*80)
    print("Deploying Gradio App to Hugging Face Spaces")
    print("="*80)

    # --- User Configuration ---
    HF_USERNAME = "LOOFYYLO"
    HF_SPACE_NAME = "interactive-futures-model"

    # Securely get the token from an environment variable
    HF_TOKEN = os.getenv("HF_TOKEN")

    if HF_USERNAME == "your-username" or not HF_TOKEN:
        print("\n⚠️  Please configure your Hugging Face username and token.")
        print("   - Set HF_USERNAME in this script.")
        print("   - Set the HF_TOKEN environment variable with your write token.")
    else:
        deploy_to_huggingface_space(
            repo_name=HF_SPACE_NAME,
            username=HF_USERNAME,
            token=HF_TOKEN,
            folder_path='app' # The folder containing our app files
        )

    print("\n" + "="*80)
