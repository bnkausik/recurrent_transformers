# Configuration for training
training_config = {
    "batch_size": 120,
    "learning_rate": 5.e-4,
    "num_steps": 5000,
    "log_interval": 100,
    "decay_flag": False
}


class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 256
    n_layer: int = 1
    n_head: int = 4
    n_embd: int = 32
    dropout: float = 0.05
    bias: bool = False
    w: int = 32  #block size for recurrence, if 0 regular transformer




