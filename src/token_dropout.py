import torch
import torch.nn as nn

class TokenImportanceDropout(nn.Module):
    """
    Applies dropout based on token importance scores.
    Importance can be calculated using various methods (gradient norm, entropy, attention mass).
    Dropout is applied before MHA/FFN by zeroing out embeddings.
    """
    def __init__(self, drop_prob, method='random', schedule='constant'): # Changed default method
        super().__init__()
        self.drop_prob = drop_prob
        # Ensure method is valid
        valid_methods = ['gradient_norm', 'entropy', 'attention_mass', 'random']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from {valid_methods}")
        self.method = method
        self.schedule = schedule # e.g., 'constant', 'linear_decay'
        # TODO: Implement schedule logic if needed

    def calculate_importance(self, embeddings, logits=None, attention_weights=None):
        """
        Calculates token importance scores based on the chosen method.
        Requires gradients for 'gradient_norm'.
        Requires logits for 'entropy'.
        Requires attention weights for 'attention_mass'.
        Returns random scores if method is 'random'.
        """
        if self.method == 'random':
            # Return random scores, shape matching (batch, seq_len)
            batch_size, seq_len, _ = embeddings.shape
            return torch.rand(batch_size, seq_len, device=embeddings.device)
        elif self.method == 'gradient_norm':
            if embeddings.grad is None:
                # Don't raise error here, handle in forward pass where retain_grad is managed
                # Return None or zeros? Let's return zeros and let forward handle it.
                # Or better, let forward pass handle the check before calling this.
                # The ValueError should be raised closer to the point of failure.
                # Let's keep the check here for clarity of requirement.
                 raise ValueError("Gradients required for 'gradient_norm' method. Ensure embeddings.retain_grad() was called and backward() was run.")
            # Ensure grad has the same shape as embeddings for norm calculation
            grad_norm = torch.norm(embeddings.grad.detach(), dim=-1) # Shape: (batch, seq_len)
            return grad_norm
        elif self.method == 'entropy':
            if logits is None:
                raise ValueError("Logits required for 'entropy' method.")
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1) # Shape: (batch, seq_len)
            # Higher entropy means more uncertainty, potentially less important? Or inverse? Decide interpretation.
            # Let's assume higher entropy -> less important for now.
            return -entropy # Negative entropy as importance score
        elif self.method == 'attention_mass':
            # Placeholder: Requires specific attention implementation details
            # Example: Sum attention weights focusing on a token across heads/layers
            if attention_weights is None:
                 raise ValueError("Attention weights required for 'attention_mass' method.")
            # This is highly dependent on how attention weights are structured/passed
            # Assuming attention_weights shape might be (batch, num_heads, seq_len, seq_len)
            # Importance could be the sum of attention received by each token
            importance = attention_weights.sum(dim=(1, 2)) # Sum over heads and source tokens -> (batch, seq_len)
            return importance
        else:
            raise ValueError(f"Unknown importance calculation method: {self.method}")

    def get_current_drop_prob(self, current_step=None, total_steps=None):
        """Returns the current dropout probability based on the schedule."""
        if self.schedule == 'linear_decay' and current_step is not None and total_steps is not None:
            return self.drop_prob * max(0, 1 - (current_step / total_steps))
        else:
            return self.drop_prob

    def get_dropout_mask(self, importance_scores, current_step=None, total_steps=None):
        """
        Creates a dropout mask based on importance scores and dropout probability.
        Lower importance scores are more likely to be dropped.
        """
        batch_size, seq_len = importance_scores.shape
        
        # Adjust drop_prob based on schedule
        current_drop_prob = self.get_current_drop_prob(current_step, total_steps)

        if current_drop_prob == 0:
            return torch.ones_like(importance_scores, dtype=torch.bool) # Keep all tokens

        # Number of tokens to keep
        num_to_keep = int(seq_len * (1 - current_drop_prob))
        if num_to_keep == seq_len:
             return torch.ones_like(importance_scores, dtype=torch.bool) # Keep all tokens
        if num_to_keep == 0:
             return torch.zeros_like(importance_scores, dtype=torch.bool) # Drop all tokens


        # Find indices of tokens to keep (those with highest importance)
        # Add small noise to break ties randomly
        noise = torch.randn_like(importance_scores) * 1e-5
        noisy_importance = importance_scores + noise
        
        # Sort importance scores and get indices
        _, sorted_indices = torch.sort(noisy_importance, dim=-1, descending=True)

        # Indices to keep
        keep_indices = sorted_indices[:, :num_to_keep]

        # Create mask
        mask = torch.zeros_like(importance_scores, dtype=torch.bool)
        mask.scatter_(dim=-1, index=keep_indices, value=True)

        return mask # True where tokens should be kept

    def forward(self, embeddings, logits=None, attention_weights=None, current_step=None, total_steps=None):
        """
        Applies token importance dropout.
        Retains gradients if needed for importance calculation.
        """
        if not self.training or self.drop_prob == 0:
            return embeddings # No dropout during eval or if drop_prob is 0

        # Handle gradient_norm requirement check more directly here
        if self.method == 'gradient_norm':
             if embeddings.grad is None:
                 raise ValueError("Cannot use 'gradient_norm' method: embeddings.grad is None. Ensure retain_grad() was called and loss.backward() was executed before this module.")
             # No need to manage requires_grad here if the training loop handles it correctly
             # (e.g., by calling retain_grad before the main forward pass if grads are needed later)

        # Calculate or generate importance scores
        importance_scores = self.calculate_importance(embeddings, logits, attention_weights) # Will be random if method='random'

        # Get mask based on scores
        dropout_mask = self.get_dropout_mask(importance_scores, current_step, total_steps)

        # Apply mask: zero out embeddings for dropped tokens
        # Ensure mask is on the same device
        dropout_mask = dropout_mask.to(embeddings.device)
        # Mask shape: (batch, seq_len), Embeddings shape: (batch, seq_len, dim)
        masked_embeddings = embeddings * dropout_mask.unsqueeze(-1).float()

        # Restore original requires_grad state if changed
        # No need to restore requires_grad if we don't modify it here

        masked_embeddings = embeddings * dropout_mask.unsqueeze(-1).float()

        return masked_embeddings

# Example Usage (conceptual)
# model = ...
# tid_dropout = TokenImportanceDropout(drop_prob=0.1, method='gradient_norm')
#
# # --- Training Loop ---
# optimizer.zero_grad()
# embeddings = model.get_embeddings(input_ids)
# embeddings.retain_grad() # IMPORTANT for gradient_norm
# logits = model(embeddings) # Assuming model takes embeddings directly after input layer
# loss = criterion(logits, targets)
# loss.backward() # Calculate gradients
#
# # Apply TID *after* backward pass but *before* optimizer step
# # Note: This modifies embeddings *after* grads were computed based on original embeddings.
# # This might be okay, or might require applying TID *before* the main model pass.
# # Let's follow the description: "dropout applied *before* MHA/FFN"
# # This implies modifying embeddings *before* they go into transformer blocks.
#
# # --- Revised Approach (applying before MHA/FFN) ---
# optimizer.zero_grad()
# base_embeddings = model.get_embeddings(input_ids) # e.g., token + pos embeddings
#
# # Option 1: Calculate importance based on a forward pass *without* dropout first
# base_embeddings_clone = base_embeddings.detach().clone()
# base_embeddings_clone.requires_grad_(True)
# logits_for_importance = model.forward_transformer_blocks(base_embeddings_clone) # Pass through model
# loss_for_importance = criterion(logits_for_importance, targets)
# loss_for_importance.backward() # Calculate grads on the clone
#
# # Now apply dropout using grads from the clone
# tid_dropout = TokenImportanceDropout(drop_prob=0.1, method='gradient_norm')
# # We need the .grad from base_embeddings_clone
# importance_scores = tid_dropout.calculate_importance(base_embeddings_clone)
# dropout_mask = tid_dropout.get_dropout_mask(importance_scores)
# masked_embeddings = base_embeddings * dropout_mask.unsqueeze(-1).float()
#
# # Now do the *actual* forward pass with the masked embeddings
# logits = model.forward_transformer_blocks(masked_embeddings) # Pass masked embeddings through model
# loss = criterion(logits, targets)
# loss.backward() # Calculate gradients for optimization based on the *masked* pass
# optimizer.step()
#
# # This seems complex. Let's stick to the simpler interpretation first:
# # Calculate importance, then apply dropout mask. The exact integration point
# # (before/after grad calc) needs careful consideration based on desired effect.
# # The description "dropout applied *before* MHA/FFN" suggests modifying the input
# # to the transformer blocks.