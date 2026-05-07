# Example config using compute-optimal scaling.
# All model and training hyperparameters are derived from depth.
#
# Usage:
#   python train.py config/train_scaled_example.py --dataset=your_dataset
#
# You can override any derived parameter after the globals().update() call.
# For example, to use a specific batch size:
#   python train.py config/train_scaled_example.py --batch_size=8

from scaling import compute_optimal_config

# The single knob: number of transformer layers
depth = 16

# Derive everything else
globals().update(compute_optimal_config(
    depth=depth,
    # vocab_size is auto-detected from dataset metadata by train.py,
    # but you can set it here for more accurate param estimates:
    # vocab_size=16384,
    block_size=1024,
    head_dim=64,         # dimension per attention head
    aspect_ratio=64,     # model_dim ~= depth * aspect_ratio
    target_param_data_ratio=20,  # Chinchilla-optimal
))

# Override anything you like after scaling:
out_dir = 'out-scaled'
data_base_dir = 'projects/paper/data'
dataset = 'openwebtext'
eval_interval = 1000
eval_iters = 200
log_interval = 10
wandb_log = False
