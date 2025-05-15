import torch
import torch.nn as nn

config = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


class AdamLLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[
                nn.DummyTransformerBlock(config)
                for _ in range(config["n_layers"])
            ]
        )
        self.final_norm = DummyLayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"],
            config["vocab_size"],
            bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeds = self.token_embedding(in_idx)
        position_embeds = self.position_embedding(
            torch.arange(seq_len),
            device=in_idx.device
        )
        x = token_embeds + position_embeds
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x
