import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # (L, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: (d_model // 2)])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, Lq, d_model)
        k: torch.Tensor,  # (B, Lk, d_model)
        v: torch.Tensor,  # (B, Lv, d_model)
        attn_mask: Optional[torch.Tensor] = None,  # broadcastable to (B, nH, Lq, Lk)
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        Lk = k.size(1)

        q = self.q_proj(q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(k).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(v).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.o_proj(out)
        return self.proj_drop(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, attn_mask=src_key_padding_mask)
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B, Lt, d_model)
        mem: torch.Tensor,  # (B, Ls, d_model)
        tgt_mask: Optional[torch.Tensor],  # (B, 1, Lt, Lt)
        src_key_padding_mask: Optional[torch.Tensor],  # (B, 1, 1, Ls)
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.drop(self_attn_out))
        cross_attn_out = self.cross_attn(x, mem, mem, attn_mask=src_key_padding_mask)
        x = self.norm2(x + self.drop(cross_attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.drop(ff_out))
        return x


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 2,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        tie_embeddings: bool = True,
        max_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        if tie_embeddings:
            self.tgt_embed.weight = self.src_embed.weight

        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_decoder_layers)]
        )

        self.norm_enc = nn.LayerNorm(d_model)
        self.norm_dec = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)
        return mask

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        B, L = tgt.shape
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(L, L, device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        return pad_mask & causal_mask

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        mask = self.make_src_key_padding_mask(src)
        for layer in self.enc_layers:
            x = layer(x, mask)
        return self.norm_enc(x)

    def decode(self, tgt: torch.Tensor, mem: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        tgt_mask = self.make_tgt_mask(tgt)
        for layer in self.dec_layers:
            x = layer(x, mem, tgt_mask, src_mask)
        return self.norm_dec(x)

    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        mem = self.encode(src)
        src_mask = self.make_src_key_padding_mask(src)
        dec_out = self.decode(tgt_inp, mem, src_mask)
        logits = self.generator(dec_out)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, sos_id: int, eos_id: int, max_len: int = 128) -> torch.Tensor:
        device = src.device
        mem = self.encode(src)
        src_mask = self.make_src_key_padding_mask(src)

        B = src.size(0)
        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            dec_out = self.decode(ys, mem, src_mask)
            logits = self.generator(dec_out[:, -1:, :])
            next_tok = torch.argmax(logits, dim=-1)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break
        return ys
