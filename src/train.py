import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
import logging
import math
import csv
import json
from itertools import islice

# Import project modules
try:
    import config
    from model import MinimalGPT
    from load_data import get_tokenizer, tokenize_and_chunk_data
    from token_dropout import TokenImportanceDropout
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure config.py, model.py, load_data.py, and token_dropout.py are in the src directory.")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
# TODO: Add file logging to logs/ directory

# --- Global Settings ---
# Enable TF32 for matmuls on Ampere+ GPUs for better performance
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    logging.info("Enabling TF32 matmul precision.")
    torch.set_float32_matmul_precision('high')
else:
    logging.info("TF32 not available or not applicable.")


def calculate_bpc(loss):
    """Calculates Bits Per Character from cross-entropy loss."""
    # Loss is NLL in base e. Convert to base 2.
    # BPC = NLL / log(2)
    # Assumes loss is averaged over tokens in the batch.
    return loss / math.log(2)

def calculate_perplexity(loss):
    """Calculates perplexity from cross-entropy loss."""
    # Perplexity = exp(loss)
    return math.exp(loss)

def evaluate(model, tokenizer, device, val_data_path, batch_size=8):
    """Evaluates model on validation data."""
    model.eval()
    val_generator = tokenize_and_chunk_data(
        val_data_path, tokenizer, config.CONTEXT_LEN + 1
    )
    
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    
    with torch.no_grad():
        while batch_count < 100:  # Evaluate on 100 batches or until data runs out
            batch_tokens = list(islice(val_generator, batch_size))
            if not batch_tokens:
                break  # No more data
                
            batch = torch.stack(batch_tokens).to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            with autocast(enabled=config.USE_MIXED_PRECISION):
                logits, loss = model(input_ids, targets=targets)
                if loss is not None:
                    total_loss += loss.item() * input_ids.size(0)  # Weighted by batch size
                    total_tokens += input_ids.numel()
                    batch_count += 1
    
    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    bpc = calculate_bpc(avg_loss)
    perplexity = calculate_perplexity(avg_loss)
    
    model.train()  # Return to training mode
    return avg_loss, bpc, perplexity

def train(experiment_name="baseline"):
    """Main training loop with experiment tracking."""
    logging.info(f"--- Starting Training: {experiment_name} ---")
    
    # Create experiment log file
    log_file = os.path.join("logs", f"{experiment_name}.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'step', 'train_loss', 'train_bpc', 'train_perplexity', 
            'val_loss', 'val_bpc', 'val_perplexity', 'learning_rate', 
            'tokens_per_sec', 'dropout_rate'
        ])

    # --- Initialization ---
    logging.info("Initializing tokenizer, model, optimizer, scheduler...")
    tokenizer = get_tokenizer()
    if tokenizer is None:
        logging.error("Failed to initialize tokenizer. Exiting.")
        return
    assert tokenizer.n_vocab == config.VOCAB_SIZE, \
        f"Tokenizer vocab size ({tokenizer.n_vocab}) doesn't match config ({config.VOCAB_SIZE})"

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = MinimalGPT(
        model_dim=config.MODEL_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        vocab_size=config.VOCAB_SIZE,
        context_len=config.CONTEXT_LEN
    ).to(device)
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scaler = GradScaler(enabled=config.USE_MIXED_PRECISION)

    tid_dropout_module = None
    if config.USE_TOKEN_DROPOUT:
        # TODO: Determine actual drop probability schedule/value
        initial_drop_prob = 0.1 # Placeholder value
        tid_dropout_module = TokenImportanceDropout(
            drop_prob=initial_drop_prob,
            method='random', # Use random dropout as gradient_norm requires complex loop changes
            # method='gradient_norm', # Requires restructuring train loop
            # method='entropy', # Requires restructuring train loop/model pass
            schedule=config.DROP_SCHEDULE
        ).to(device)
        logging.info(f"Token Importance Dropout enabled (Method: {tid_dropout_module.method}, Schedule: {tid_dropout_module.schedule})")


    # --- Data Loading ---
    logging.info("Setting up data loaders...")
    # Use the generator directly for streaming-like behavior
    train_data_generator = tokenize_and_chunk_data(
        config.DATA_PATH, tokenizer, config.CONTEXT_LEN + 1 # Need context + 1 for target
    )
    # Note: val data loading would be similar, likely in a separate eval function

    # --- Training Loop ---
    model.train()
    start_time = time.time()
    tokens_processed = 0
    accumulated_loss = 0.0
    log_interval = 100 # Log every N steps

    # Simple warmup implementation
    def get_lr(step):
        if step < config.WARMUP_STEPS:
            # Cosine warmup instead of linear
            return config.LR * 0.5 * (1 + math.cos(math.pi * (1 - step/config.WARMUP_STEPS)))
        else:
            # Cosine decay with minimum LR at 10% of max
            progress = (step - config.WARMUP_STEPS) / (config.TOTAL_STEPS - config.WARMUP_STEPS)
            return config.LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    logging.info(f"Starting training for {config.TOTAL_STEPS} steps...")
    for step in range(config.TOTAL_STEPS):
        # Adjust LR based on warmup or scheduler
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Get batch data
        batch_tokens = list(islice(train_data_generator, config.BATCH_SIZE))
        if not batch_tokens:
            logging.warning("Data generator exhausted. Resetting generator.")
            # Reset generator if it runs out (assuming repeatable dataset)
            train_data_generator = tokenize_and_chunk_data(
                config.DATA_PATH, tokenizer, config.CONTEXT_LEN + 1
            )
            batch_tokens = list(islice(train_data_generator, config.BATCH_SIZE))
            if not batch_tokens:
                 logging.error("Failed to get data even after resetting generator. Exiting.")
                 break # Exit if data is truly exhausted or empty

        # Stack tensors and prepare input/target
        batch = torch.stack(batch_tokens).to(device)
        input_ids = batch[:, :-1] # Input: first context_len tokens
        targets = batch[:, 1:]   # Target: last context_len tokens (shifted)

        # --- Forward and Backward Pass ---
        optimizer.zero_grad()

        # Mixed Precision Context
        with autocast(enabled=config.USE_MIXED_PRECISION):
            # --- TID Integration ---
            # This is tricky, especially for gradient_norm.
            # If using gradient_norm, we might need to:
            # 1. Initial forward/backward pass to get gradients on embeddings.
            # 2. Apply TID based on those gradients.
            # 3. Second forward/backward pass with masked embeddings for the actual update.
            # This doubles computation. Let's use a simpler placeholder for now.
            # The model's forward pass now accepts the dropout module.
            logits, loss = model(
                input_ids,
                targets=targets,
                token_dropout_module=tid_dropout_module,
                current_step=step,
                total_steps=config.TOTAL_STEPS
                )

            if loss is None:
                logging.error("Loss calculation failed.")
                continue # Skip step if loss is None

        # Scale loss and backward pass
        scaler.scale(loss).backward()

        # Gradient Clipping (optional but recommended)
        scaler.unscale_(optimizer) # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # --- Logging ---
        accumulated_loss += loss.item()
        tokens_processed += input_ids.numel()

        if step % log_interval == 0 and step > 0:
            avg_loss = accumulated_loss / log_interval
            bpc = calculate_bpc(avg_loss)
            perplexity = calculate_perplexity(avg_loss)
            elapsed_time = time.time() - start_time
            steps_per_sec = log_interval / elapsed_time
            tokens_per_sec = tokens_processed / elapsed_time
            current_lr = optimizer.param_groups[0]['lr'] # Get current LR
            
            # Get dropout rate if available
            dropout_rate = tid_dropout_module.get_current_drop_prob(step, config.TOTAL_STEPS) if tid_dropout_module else 0.0

            logging.info(
                f"Step: {step}/{config.TOTAL_STEPS} | "
                f"Loss: {avg_loss:.4f} | "
                f"BPC: {bpc:.4f} | "
                f"PPL: {perplexity:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Steps/sec: {steps_per_sec:.2f} | "
                f"Tokens/sec: {tokens_per_sec:.0f}"
            )
            
            # Log to CSV
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, avg_loss, bpc, perplexity, 
                    None, None, None,  # Val metrics will be added during evaluation
                    current_lr, tokens_per_sec, dropout_rate
                ])

            # Reset counters for next interval
            accumulated_loss = 0.0
            tokens_processed = 0
            start_time = time.time()

        # --- Periodic Evaluation and Checkpointing (Placeholders) ---
        eval_interval = 1000 # Evaluate every N steps
        checkpoint_interval = 5000 # Save checkpoint every N steps

        if step % eval_interval == 0 and step > 0:
            logging.info(f"--- Running Evaluation at Step {step} ---")
            val_loss, val_bpc, val_perplexity = evaluate(
                model, tokenizer, device, config.VAL_PATH, batch_size=config.BATCH_SIZE
            )
            logging.info(
                f"Validation: Loss: {val_loss:.4f} | "
                f"BPC: {val_bpc:.4f} | "
                f"Perplexity: {val_perplexity:.4f}"
            )
            
            # Update CSV with validation metrics
            with open(log_file, 'r', newline='') as f:
                rows = list(csv.reader(f))
            
            # Find last entry with matching step and update it
            for i in range(len(rows)-1, -1, -1):
                if len(rows[i]) > 0 and rows[i][0] == str(step):
                    rows[i][4] = val_loss
                    rows[i][5] = val_bpc
                    rows[i][6] = val_perplexity
                    break
            
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            
            model.train() # Ensure model is back in training mode

        if step % checkpoint_interval == 0 and step > 0:
            logging.info(f"--- Saving Checkpoint at Step {step} ---")
            checkpoint_path = os.path.join("checkpoints", f"model_step_{step}.pt")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss, # Save last logged average loss
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
            # TODO: Save best model based on validation loss

    logging.info("--- Training Finished ---")
    # Save final model
    final_checkpoint_path = os.path.join("checkpoints", "model_final.pt")
    torch.save({
        'step': config.TOTAL_STEPS,
        'model_state_dict': model.state_dict(),
        # Add other relevant states if needed
    }, final_checkpoint_path)
    logging.info(f"Final model saved to {final_checkpoint_path}")


# Placeholder for evaluation function (to be implemented in evaluate.py)
# def evaluate(model, tokenizer, device):
#     # Load validation data
#     # Set model to eval mode
#     # Run inference, calculate val loss and BPC
#     # Log results
#     pass

if __name__ == "__main__":
    import sys
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    train(experiment_name)
