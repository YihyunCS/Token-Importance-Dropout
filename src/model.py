import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# Import config values (assuming config.py is in the same directory or accessible)
try:
    import config
except ImportError:
    # Define defaults if config.py is not found (e.g., during direct execution)
    # This is not ideal for a real project but helps make the file runnable standalone
    print("Warning: config.py not found. Using default model parameters.")
    @dataclass
    class MockConfig:
        MODEL_DIM: int = 768
        NUM_LAYERS: int = 12
        NUM_HEADS: int = 12
        VOCAB_SIZE: int = 50257
        CONTEXT_LEN: int = 128
        DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
        # Add other necessary defaults if used directly in model.py
    config = MockConfig()


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embedding to the input tensor."""
    # Reshape input to complex numbers
    x_ = x.float().reshape(*x.shape[:-1], -1, 2) # (..., seq_len, heads * dim_per_head//2, 2)
    
    # Handle freqs_cis - it's already complex from precompute_freqs_cis()
    if freqs_cis.is_complex():
        # If already complex, just reshape for broadcasting
        freqs_cis = freqs_cis.unsqueeze(1) # (seq_len, 1, dim_per_head//2)
    else:
        # If not complex (legacy case), convert to complex
        freqs_cis = freqs_cis.view(x_.shape[1], 1, -1, 2) # Reshape to (seq_len, 1, dim_per_head//2, 2)
        freqs_cis = torch.view_as_complex(freqs_cis)

    # Convert input to complex and apply rotation
    x_out = torch.view_as_complex(x_)
    x_rotated = x_out * freqs_cis
    
    # Convert back to real and flatten
    return torch.view_as_real(x_rotated).flatten(3).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes the frequency tensor for rotary embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class Attention(nn.Module):
    """Multi-Head Attention module with RoPE and no bias."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor | None = None):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Transpose for attention calculation: (batch_size, num_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # Add mask (e.g., for causal attention)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU FeedForward module."""
    def __init__(self, dim: int, hidden_dim: int | None = None, multiple_of: int = 256):
        super().__init__()
        if hidden_dim is None:
            # Standard hidden dim calculation for SwiGLU, often 2/3 * 4 * dim
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # Gate proj
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # Up proj
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # Down proj

    def forward(self, x):
        # SwiGLU activation: silu(w1(x)) * w3(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """A single transformer block with RMSNorm, Attention, and FeedForward."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attention = Attention(dim, num_heads)
        self.feed_forward = FeedForward(dim=dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor | None):
        # Pre-normalization
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class MinimalGPT(nn.Module):
    """Minimal GPT-2 variant with RoPE, RMSNorm, SwiGLU, no bias."""
    def __init__(self, model_dim: int, num_layers: int, num_heads: int, vocab_size: int, context_len: int):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.context_len = context_len

        self.tok_embeddings = nn.Embedding(vocab_size, model_dim)
        # Note: No explicit position embeddings needed due to RoPE

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(model_dim, num_heads))

        self.norm = RMSNorm(model_dim) # Final normalization
        self.output = nn.Linear(model_dim, vocab_size, bias=False) # LM head

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            self.model_dim // self.num_heads, # RoPE applied per head
            self.context_len * 2 # Allow for longer sequences? Or just context_len? Let's use context_len for now.
        )

        # Optional: Weight tying
        # self.tok_embeddings.weight = self.output.weight

        # Initialize weights (example)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: # Should not happen with bias=False
                 torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None, token_dropout_module: nn.Module | None = None, current_step=None, total_steps=None):
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.context_len, f"Input sequence length ({seq_len}) exceeds context length ({self.context_len})"

        # Ensure freqs_cis is on the correct device
        self.freqs_cis = self.freqs_cis.to(tokens.device)
        freqs_cis = self.freqs_cis[:seq_len] # Slice for current sequence length

        h = self.tok_embeddings(tokens) # (batch, seq_len, dim)

        # --- Token Importance Dropout Integration Point ---
        # Apply *before* transformer blocks as per required.md
        if token_dropout_module is not None and self.training:
             # We need importance scores. How to get them?
             # Option A: Pass gradients if method='gradient_norm' (requires backward pass first - complex)
             # Option B: Pass logits if method='entropy' (requires a forward pass first - complex)
             # Option C: Pass attention weights if method='attention_mass' (requires modification to Attention block)
             # Option D: Simpler approach - apply dropout randomly or based on magnitude (not specified in required.md)

             # Let's assume for now the dropout module handles its own importance calculation
             # or that importance scores are passed in somehow.
             # If using gradient norm, the training loop needs modification (see token_dropout.py comments)
             # If using entropy, we might need a preliminary forward pass.

             # Simplest integration: Apply the module directly to embeddings.
             # The module itself needs to figure out importance.
             # If method='gradient_norm', it might need grads attached to 'h'.
             if token_dropout_module.method == 'gradient_norm':
                 h.retain_grad() # Signal that we need grads for these embeddings

             # Pass necessary info for schedule calculation
             h = token_dropout_module(h, current_step=current_step, total_steps=total_steps)
             # Note: If using gradient_norm, the actual importance calculation and masking
             # might need to happen *after* a backward pass in the training loop.
             # The current placement applies dropout *before* the blocks based on potentially
             # stale or unavailable importance scores depending on the method.
             # This needs careful handling in the training script.

        # --- Transformer Blocks ---
        # Create causal mask
        mask = None
        if seq_len > 1:
            mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).type_as(h) # Upper triangular part is masked

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h) # Final normalization

        loss = None
        if targets is not None:
            # Calculate loss if targets are provided
            logits = self.output(h) # (batch, seq_len, vocab_size)
            # Ensure tensors are contiguous before reshaping
            logits = logits.contiguous()
            targets = targets.contiguous()
            # Reshape for CrossEntropyLoss: (batch * seq_len, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # ignore_index=-1 is common practice for padding tokens if used
        else:
            # Only compute logits for the last token during inference/generation
            logits = self.output(h[:, [-1], :]) # (batch, 1, vocab_size)


        return logits, loss

    def get_embeddings(self, tokens: torch.Tensor):
        """Helper to get initial embeddings, potentially useful for TID."""
        return self.tok_embeddings(tokens)

    def forward_transformer_blocks(self, h: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor | None):
         """Helper to forward through transformer blocks only, useful for complex TID."""
         for layer in self.layers:
            h = layer(h, freqs_cis, mask)
         h = self.norm(h)
         logits = self.output(h)
         return logits


# --- Global Settings ---
# Enable TF32 for matmuls on Ampere+ GPUs
# Do this in the main script (train.py) for clarity
# torch.set_float32_matmul_precision('high')

# --- Example Usage (in train.py) ---
# model = MinimalGPT(
#     model_dim=config.MODEL_DIM,
#     num_layers=config.NUM_LAYERS,
#     num_heads=config.NUM_HEADS,
#     vocab_size=config.VOCAB_SIZE,
#     context_len=config.CONTEXT_LEN
# ).to(config.DEVICE)
#
# # Mixed Precision (in training loop)
# from torch.cuda.amp import autocast
# with autocast(enabled=config.USE_MIXED_PRECISION):
#     logits, loss = model(input_ids, targets)
#
# # TF32 enabled globally before training starts
# torch.set_float32_matmul_precision('high')