"""
(legacy script)
Generate next-token prediction probabilities for each position in an input text.

Takes a small input text string and outputs a CSV with columns:
- position: the position in the sequence (0-indexed)
- target: the actual token at that position
- One column per vocabulary token containing the predicted probability

This is similar to build_fcm.py with include_target=True and window=0, 
but without aggregation - each row is a single position's predictions.
"""
import os
import pickle
import csv
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default configuration (can be overridden via command line or config file)
# -----------------------------------------------------------------------------

# Model loading
init_from = 'resume'  # 'resume' (from out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'

# Input text
input_text = ''  # the text to process (required)
input_file = ''  # alternatively, read from a file

# Processing
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False  # use PyTorch 2.0 compilation
seed = 1337

# Output
output_file = 'predictions.csv'  # output CSV file

# Encoding/decoding config
sep = '' # only supported for meta.pkl tokenizers

exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------


def main():
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup device and dtype
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Load model
    print("Loading model...")
    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    else:
        raise ValueError(f"Unknown init_from: {init_from}")
    
    model.eval()
    model.to(device)
    if compile_model:
        model = torch.compile(model)
    
    vocab_size = model.config.vocab_size
    block_size = model.config.block_size
    print(f"Model vocab size: {vocab_size}, block size: {block_size}")
    
    # Load tokenizer/encoder
    encode = None
    decode = None
    itos = None
    
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_pickle_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        meta_json_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.json')
        
        if os.path.exists(meta_pickle_path):
            print(f"Loading meta from {meta_pickle_path}...")
            with open(meta_pickle_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s.split(sep)]
            decode = lambda l: sep.join([itos[str(i)] for i in l])
        elif os.path.exists(meta_json_path):
            from tokenizers import Tokenizer
            print(f"Loading meta from {meta_json_path}...")
            tokenizer = Tokenizer.from_file(meta_json_path)
            encode = lambda s: tokenizer.encode(s).ids
            decode = lambda l: tokenizer.decode(l, skip_special_tokens=False)
            # Build itos from tokenizer vocab
            itos = {i: tokenizer.id_to_token(i) for i in range(vocab_size)}
    
    if encode is None:
        import tiktoken
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        # Build itos from tiktoken
        itos = {i: enc.decode([i]) for i in range(vocab_size)}
    
    
    # Get input text
    text = input_text
    if input_file:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    if not text:
        raise ValueError("No input text provided. Set input_text='...' or input_file='path/to/file.txt'")
    
    # Encode input
    tokens = encode(text)
    n_tokens = len(tokens)
    print(f"Input text: {repr(text[:100])}{'...' if len(text) > 100 else ''}")
    print(f"Encoded to {n_tokens} tokens")
    
    if n_tokens == 0:
        raise ValueError("Input text encoded to zero tokens")
    
    if n_tokens > block_size:
        print(f"Warning: input ({n_tokens} tokens) exceeds block_size ({block_size}). "
              f"Later positions will use truncated context.")
    
    # Prepare column headers
    # Use token strings if available, otherwise just indices
    token_columns = [decode([i]) for i in range(vocab_size)] if decode else [i for i in range(vocab_size)]
    headers = ['.position', '.target'] + token_columns
    
    # Process each position and write to CSV
    print(f"\nGenerating predictions for {n_tokens} positions...")
    print(f"Output file: {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        with torch.no_grad():
            with ctx:
                for pos in range(n_tokens):
                    # Context is all tokens up to (but not including) this position
                    # For position 0, we have no context - use empty or handle specially
                    if pos == 0:
                        # No context available for position 0
                        # Use uniform distribution or skip
                        context = torch.zeros((1, 1), dtype=torch.long, device=device)
                        # We'll still run the model but the prediction won't be meaningful
                        # Some models expect at least one token, so we use token 0
                        context[0, 0] = tokens[0] if tokens else 0
                    else:
                        # Use context from start (or truncated if too long)
                        ctx_start = max(0, pos - block_size + 1)
                        context_tokens = tokens[ctx_start:pos]
                        context = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
                    
                    # Forward pass
                    logits, _ = model(context)
                    probs = F.softmax(logits[0, -1, :], dim=-1).float().cpu().numpy()
                    
                    # Get target token (the actual token at this position)
                    target_id = tokens[pos]
                    target_str = decode([target_id]) if decode else str(target_id)
                    
                    # Write row
                    row = [pos, target_str] + [f'{p:.6e}' for p in probs]
                    writer.writerow(row)
                    
                    if (pos + 1) % 100 == 0 or pos == n_tokens - 1:
                        print(f"  Processed {pos + 1}/{n_tokens} positions")
    
    print(f"\nDone! Output written to {output_file}")
    print(f"CSV dimensions: {n_tokens} rows x {len(headers)} columns")


if __name__ == '__main__':
    main()
