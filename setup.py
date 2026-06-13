import subprocess
import sys

# The exact libraries your deepfake defense app needs
libraries = [
    "tensorflow",
    "streamlit",
    "opencv-python",
    "mtcnn",
    "numpy",
    "Pillow"
]

print(f"Forcing installation into Python environment:\n{sys.executable}\n")

for lib in libraries:
    print(f"Installing {lib}...")
    try:
        # This calls pip directly from the Python interpreter
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print(f"DONE: {lib} installed successfully!\n")
    except subprocess.CalledProcessError:
        print(f"ERROR: Failed to install {lib}.\n")

print("All done! You can now run your scripts.")