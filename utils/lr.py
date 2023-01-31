import math


class LR():
    def __init__(self, lr, decay=False, warmup_tokens=1, final_tokens=1e6) -> None:
        self.lr = lr
        self.decay = decay
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens

    def step(self, trained_tokens):
        # decay the learning rate based on our progress
        if self.decay:
            if trained_tokens < self.warmup_tokens:
                # linear warmup
                lr_mult = float(trained_tokens) / float(max(1, self.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(trained_tokens - self.warmup_tokens) / float(max(1, self.final_tokens - self.warmup_tokens))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            self.lr *= lr_mult