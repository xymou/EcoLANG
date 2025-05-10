from transformers import LogitsProcessor
import torch

class VocabLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Modifies the logits so that only tokens in `allowed_token_ids` are considered.
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        # Create a mask for the allowed tokens
        mask = torch.full_like(logits, float('-inf'))  # start with a tensor of -inf
        mask[:, list(self.allowed_token_ids)] = 0  # set logits for allowed tokens to 0

        # Add the mask to the logits: this will set logits of non-allowed tokens to -inf
        logits = logits + mask

        return logits