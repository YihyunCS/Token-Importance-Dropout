import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import os
import logging
import math
from itertools import islice
import argparse

# Import project modules
try:
    import config
    from model import MinimalGPT
    from load_data import get_tokenizer, tokenize_and_chunk_data
    # TokenImportanceDropout is not needed here as it's disabled during eval
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure config.py, model.py, and load_data.py are in the src directory.")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_bpc(loss):
    """Calculates Bits Per Character from cross-entropy loss."""
    return loss / math.log(2)

def evaluate(checkpoint_path=None):
    """Evaluation loop."""
    logging.info("--- Starting Evaluation ---")

    # --- Initialization ---
    logging.info("Initializing tokenizer and model...")
    tokenizer = get_tokenizer()
    if tokenizer is None:
        logging.error("Failed to initialize tokenizer. Exiting.")
        return
    assert tokenizer.n_vocab == config.VOCAB_SIZE, \
        f"Tokenizer vocab size ({tokenizer.n_vocab}) doesn't match config ({config.VOCAB_SIZE})"

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize model architecture
    model = MinimalGPT(
        model_dim=config.MODEL_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        vocab_size=config.VOCAB_SIZE,
        context_len=config.CONTEXT_LEN
    ).to(device)

    # --- Load Checkpoint ---
    if checkpoint_path is None:
        # Default to loading the final model if no specific path is given
        checkpoint_path = os.path.join("checkpoints", "model_final.pt")
        logging.info("No checkpoint specified, attempting to load 'checkpoints/model_final.pt'")

    if os.path.exists(checkpoint_path):
        logging.info(f"Loading model checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Handle potential DataParallel wrapping if saved that way (unlikely here)
            state_dict = checkpoint.get('model_state_dict', checkpoint) # Allow loading raw state dict too
            # Load weights, be flexible about strict matching if needed
            model.load_state_dict(state_dict, strict=True)
            step = checkpoint.get('step', 'N/A')
            logging.info(f"Successfully loaded checkpoint (Step: {step}).")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}. Proceeding with initialized weights (likely random).")
    else:
        logging.warning(f"Checkpoint file not found: {checkpoint_path}. Evaluating with initialized weights.")


    # --- Data Loading ---
    logging.info("Setting up validation data loader...")
    val_data_generator = tokenize_and_chunk_data(
        config.VAL_PATH, tokenizer, config.CONTEXT_LEN + 1 # Need context + 1 for target
    )

    # --- Evaluation Loop ---
    model.eval() # Set model to evaluation mode (disables dropout in nn.Dropout, etc.)
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    # Determine number of batches based on VAL_SAMPLES and BATCH_SIZE for a full eval
    # Or use a fixed number for quicker eval
    eval_total_samples = config.VAL_SAMPLES # Use VAL_SAMPLES from required.md (implicitly via load_data)
    eval_batches = math.ceil(eval_total_samples / config.BATCH_SIZE) # Calculate batches needed
    logging.info(f"Running evaluation on {config.VAL_PATH} ({eval_total_samples} samples, ~{eval_batches} batches)...")


    # Proper batching for evaluation
    batch_iterator = iter(val_data_generator)

    with torch.no_grad(): # Disable gradient calculations
        for i in range(eval_batches):
            # Collect a batch
            batch_sequences = list(islice(batch_iterator, config.BATCH_SIZE))
            if not batch_sequences:
                break # Stop if data runs out

            # Stack tensors and prepare input/target
            batch = torch.stack(batch_sequences).to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            # Mixed precision context
            with autocast(enabled=config.USE_MIXED_PRECISION):
                 logits, loss = model(input_ids, targets=targets) # TID module not passed

                 if loss is not None:
                     # Accumulate loss, weighted by number of tokens in the batch
                     total_loss += loss.item() * targets.numel()
                     total_tokens += targets.numel()
                     num_batches += 1 # Counting batches processed
                 else:
                      logging.warning("Loss calculation returned None for a batch.")

    if total_tokens == 0:
        logging.error("No tokens were processed during evaluation. Check data loading and batching.")
        return

    # Calculate average loss and BPC
    avg_loss = total_loss / total_tokens
    bpc = calculate_bpc(avg_loss)

    logging.info("--- Evaluation Finished ---")
    logging.info(f"Validation Loss: {avg_loss:.4f}")
    logging.info(f"Validation BPC: {bpc:.4f}")
    logging.info(f"Evaluated on {num_batches} batches ({total_tokens} tokens).")

    # TODO: Log results to file in logs/

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained MinimalGPT model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint (.pt file). Defaults to 'checkpoints/model_final.pt'."
    )
    args = parser.parse_args()
    evaluate(checkpoint_path=args.checkpoint)
