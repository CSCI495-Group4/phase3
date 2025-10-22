# data/dataloorer_text.py
import torch
import random
from utils.config import MAX_TOKENS, VOCAB_SIZE

def dummy_tokenize(batch_size):
    """
    Simulates tokenized text until you have real text.
    Returns dict with input_ids and attention_mask (B, L)
    """
    seq_len = MAX_TOKENS
    # Random token ids (excluding 0 as pad) just to exercise the text branch
    input_ids = torch.randint(low=1, high=VOCAB_SIZE, size=(batch_size, seq_len))
    # All tokens attended to (you can randomize for fun)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
