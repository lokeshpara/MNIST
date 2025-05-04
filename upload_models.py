import os
import shutil

def copy_models_to_render():
    """
    Copy model files to the models directory for Render deployment.
    This script should be run before deploying to Render.
    """
    # Source directory (where your model files are stored)
    source_dir = input("Enter the path to your model files directory: ").strip()
    
    # Destination directory (models directory in your project)
    dest_dir = "models"
    
    # Create models directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # List of model files to copy
    model_files = [
        "mnist_model.pth",
        "mnist_model_l1.pth",
        "mnist_model_l2.pth",
        "mnist_model_l1_l2.pth"
    ]
    
    # Copy each model file
    for model_file in model_files:
        source_path = os.path.join(source_dir, model_file)
        dest_path = os.path.join(dest_dir, model_file)
        
        if os.path.exists(source_path):
            print(f"Copying {model_file}...")
            shutil.copy2(source_path, dest_path)
            print(f"Successfully copied {model_file}")
        else:
            print(f"Warning: {model_file} not found in source directory")
    
    print("\nModel files have been copied to the models directory.")
    print("You can now deploy your application to Render.")

if __name__ == "__main__":
    copy_models_to_render() 