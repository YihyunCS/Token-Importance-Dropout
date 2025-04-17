import os
import json
import tiktoken
from datasets import load_dataset
from itertools import islice
import logging
import torch # Added torch import here as it's used in tokenize_and_chunk_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import config values
try:
    import config
except ImportError:
    logging.warning("config.py not found. Using default paths and parameters.")
    class MockConfig:
        DATA_PATH = "data/train.jsonl"
        VAL_PATH = "data/val.jsonl"
        CONTEXT_LEN = 128
        # Add other necessary defaults if used directly
    config = MockConfig()

# Constants
DATASET_NAME = "openwebtext"
TOKENIZER_NAME = "gpt2"
TOKENIZER_CACHE_DIR = "tokenizer"
TOKENIZER_FILE = os.path.join(TOKENIZER_CACHE_DIR, "tokenizer.json") # As per required.md
TRAIN_SAMPLES = 10_000
VAL_SAMPLES = 1_000

def download_and_limit_dataset(output_path, num_samples, split='train'):
    """
    Downloads (streams) a split of the OpenWebText dataset, limits the number of samples,
    and saves each document as a raw string in a JSONL file.
    """
    if os.path.exists(output_path):
        logging.info(f"Dataset file already exists: {output_path}. Skipping download.")
        return

    logging.info(f"Loading '{DATASET_NAME}' dataset (split: {split}) with streaming...")
    try:
        # Load the dataset in streaming mode
        streamed_dataset = load_dataset(DATASET_NAME, split=split, streaming=True)
    except Exception as e:
        logging.error(f"Failed to load dataset '{DATASET_NAME}': {e}")
        logging.error("Please ensure you have internet connectivity and the 'datasets' library installed.")
        logging.error("You might need to log in to Hugging Face Hub: `huggingface-cli login`")
        return

    logging.info(f"Taking first {num_samples} samples using islice...")
    limited_samples = islice(streamed_dataset, num_samples)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logging.info(f"Saving limited samples to {output_path}...")
    count = 0
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in limited_samples:
                # Assuming the dataset provides samples with a 'text' key
                if 'text' in sample:
                    # Write the raw text as a JSON object per line
                    json.dump({'text': sample['text']}, f)
                    f.write('\n')
                    count += 1
                else:
                    logging.warning(f"Sample missing 'text' key: {sample}")
                if count % 1000 == 0 and count > 0:
                     logging.info(f"Saved {count} samples...")

        logging.info(f"Successfully saved {count} samples to {output_path}.")
        if count < num_samples:
             logging.warning(f"Warning: Only found {count} samples, less than the requested {num_samples}.")

    except Exception as e:
        logging.error(f"Error writing to {output_path}: {e}")
        # Clean up potentially incomplete file
        if os.path.exists(output_path):
            os.remove(output_path)

def get_tokenizer(cache_dir=TOKENIZER_CACHE_DIR, tokenizer_file=TOKENIZER_FILE):
    """
    Loads the GPT-2 tokenizer using tiktoken.
    Attempts to load from a cached file first, otherwise fetches and saves.
    Note: tiktoken doesn't directly save/load like HF tokenizers' tokenizer.json.
          This function primarily ensures the encoding is available.
          The 'tokenizer.json' mentioned in required.md might be a conceptual placeholder
          or require a different library (like HuggingFace tokenizers) if strict serialization is needed.
          For tiktoken, we just need the encoding name.
    """
    logging.info(f"Loading tokenizer: {TOKENIZER_NAME}")
    try:
        enc = tiktoken.get_encoding(TOKENIZER_NAME)
        # Simulate saving/caching if needed, though tiktoken handles this internally
        os.makedirs(cache_dir, exist_ok=True)
        # We can't easily serialize a tiktoken encoder to a single JSON file
        # in the same way as HF tokenizers. We'll just return the encoder.
        # If required.md implies using HF tokenizers library for the .json,
        # this part would need to change.
        logging.info(f"Tokenizer '{TOKENIZER_NAME}' loaded successfully (Vocab size: {enc.n_vocab}).")
        # Placeholder for saving logic if needed for other reasons:
        # if not os.path.exists(tokenizer_file):
        #     logging.warning(f"Cannot directly save tiktoken encoder to {tokenizer_file}. Tiktoken manages its own cache.")
            # save_logic_here_if_possible()
        return enc
    except Exception as e:
        logging.error(f"Failed to load tokenizer '{TOKENIZER_NAME}': {e}")
        return None

def tokenize_and_chunk_data(jsonl_path, tokenizer, context_length, overlap=0):
    """
    Loads data from a JSONL file, tokenizes it, and chunks it into sequences.
    This is a generator function to handle potentially large datasets efficiently.

    Args:
        jsonl_path (str): Path to the .jsonl file containing {'text': '...'} objects.
        tokenizer: An initialized tiktoken tokenizer instance.
        context_length (int): The desired length of each chunk.
        overlap (int): Number of tokens to overlap between chunks (optional).

    Yields:
        torch.Tensor: A tensor of shape (context_length,) containing token IDs.
    """
    if not os.path.exists(jsonl_path):
        logging.error(f"Data file not found: {jsonl_path}")
        return

    logging.info(f"Tokenizing and chunking data from {jsonl_path} with context={context_length}, overlap={overlap}...")
    buffer = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    if text:
                        tokens = tokenizer.encode_ordinary(text) # Use encode_ordinary for raw text
                        buffer.extend(tokens)

                        # Yield chunks from the buffer
                        while len(buffer) >= context_length:
                            chunk = buffer[:context_length]
                            yield torch.tensor(chunk, dtype=torch.long) # Convert to tensor
                            # Move buffer forward, considering overlap
                            buffer = buffer[context_length - overlap:]

                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line in {jsonl_path}: {line.strip()}")

        # Yield any remaining part of the buffer if it's a full chunk (or handle padding if needed)
        # This simple version doesn't handle the last partial chunk.
        # Depending on training needs, you might pad or discard the remainder.
        logging.info(f"Finished processing {jsonl_path}.")

    except Exception as e:
        # Check if logging is still available during potential shutdown
        if logging:
            logging.error(f"Error processing file {jsonl_path}: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # Import torch here for the example usage in main block
    # Already imported at the top now

    logging.info("--- Starting Data Preparation ---")

    # 1. Download/Verify Raw Data
    logging.info("Step 1: Preparing Training Data...")
    download_and_limit_dataset(config.DATA_PATH, TRAIN_SAMPLES, split='train')

    logging.info("Step 2: Preparing Validation Data...")
    download_and_limit_dataset(config.VAL_PATH, VAL_SAMPLES, split='train') # OpenWebText doesn't have a standard val split, use train

    # 2. Load Tokenizer
    logging.info("Step 3: Loading Tokenizer...")
    tokenizer = get_tokenizer()

    # 3. Example: Tokenize and Chunk (Optional - often done dynamically in training)
    if tokenizer:
        logging.info("Step 4: Example of Tokenizing and Chunking (first 5 chunks of validation data)...")
        # Create a generator
        chunk_generator = tokenize_and_chunk_data(config.VAL_PATH, tokenizer, config.CONTEXT_LEN)
        # Get the first 5 chunks
        for i, chunk in enumerate(islice(chunk_generator, 5)):
            logging.info(f"Chunk {i+1} shape: {chunk.shape}, first 10 tokens: {chunk[:10].tolist()}...")
            if i >= 4: break # Limit to 5 examples
    else:
        logging.warning("Skipping tokenization example because tokenizer failed to load.")

    logging.info("--- Data Preparation Script Finished ---")
