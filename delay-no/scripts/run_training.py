#!/usr/bin/env python
"""
Clean runner script that sets environment variables to suppress warnings
before running the actual training script.
"""
import os
import sys
import subprocess

# Set environment variables to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow logging
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::DeprecationWarning'  # Ignore future and deprecation warnings
os.environ['HYDRA_FULL_ERROR'] = '1'  # Show full Hydra error traces
os.environ['TORCH_WARN_ONCE'] = '1'  # Make PyTorch only warn once for each warning

# Import PyTorch first and set precision
import torch
torch.set_float32_matmul_precision('high')  # Use Tensor Cores efficiently

# Run the actual training script
if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the actual training script
    train_script = os.path.join(current_dir, "train.py")
    
    # Forward any command-line arguments to the training script
    cmd_args = [sys.executable, train_script] + sys.argv[1:]
    
    # Run the training script
    result = subprocess.run(cmd_args)
    
    # Exit with the same code
    sys.exit(result.returncode)
