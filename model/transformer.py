import torch
import torch.nn as nn
import math
from typing import Optional
from flash_attn import flash_attn_varlen_qkvpacked_func
# -----------------------------------------------------------------------------
# Positional Encoding
# ----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self._init_pe(max_len)  # 初始化位置编码
        self.max_len = max_len

    def _init_pe(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_item = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_item)
        pe[:, 1::2] = torch.cos(position * div_item)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def _expand_pe(self, new_len: int, device):
        """如果当前 pe 不够长，动态扩展位置编码"""
        if new_len <= self.pe.size(1):
            return  # 不需要扩展

        new_pe = torch.zeros(new_len, self.d_model, device=device)
        position = torch.arange(0, new_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_item = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / self.d_model)
        )
        new_pe[:, 0::2] = torch.sin(position * div_item)
        new_pe[:, 1::2] = torch.cos(position * div_item)
        new_pe = new_pe.unsqueeze(0)  # (1, new_len, d_model)

        self.pe = new_pe  # 更新 buffer
        self.max_len = new_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        self._expand_pe(seq_len, device=x.device)
        pos_vec = self.pe[:, :seq_len, :]
        return x + pos_vec


# -----------------------------------------------------------------------------
# Multi‑Head Attention
# -----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """多头自注意力（不含 QKV 偏置，与原论文一致）。

    Args:
        d_model (int): 隐藏维度 `D`。
        num_heads (int): 头数 `H`，需满足 `d_model % num_heads == 0`。
        dropout (float, optional): 注意力权重 dropout 概率。默认 0.1。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
# ------------------------------------------------------------------
        # TODO (学生实现)：定义 Multi‑Head Attention 所需的线性层与超参数。
        # 需要的核心组件：
        #   * Q、K、V 的独立线性变换 (bias=False)
        #   * 输出线性层 ``out_proj``
        #   * dropout 层
        # ------------------------------------------------------------------

        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # TODO: 定义线性层等
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)  # 输出线性层


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        USEFLASHATTEN: bool = False,
    ) -> torch.Tensor:  # (B, L_q, D)
        """计算多头注意力。

        Args:
            query (Tensor): ``(B, L_q, D)`` 查询张量。
            key   (Tensor): ``(B, L_k, D)`` 键张量。
            value (Tensor): ``(B, L_v, D)`` 值张量。
            mask  (Tensor, optional): ``(B, 1, L_q, L_k)`` 或可广播的
                布尔张量。**1 表示可见，0 表示遮挡。**
        Returns:
            Tensor: ``(B, L_q, D)`` 经过注意力聚合后的表示。
        """
        # ------------------------------------------------------------------
        # TODO (学生实现)：完成 "Scaled Dot‑Product Attention" 计算流程。
        # 步骤顺序：
        #   1. 对 Q/K/V 做线性映射并 reshape → ``(B, H, L, head_dim)``
        #   2. 计算缩放点积得分，再根据 mask 设 -inf，softmax + dropout
        #   3. 聚合 V，最后经 ``out_proj`` 投影回原维度
        # ------------------------------------------------------------------
        B, L_q, D = query.shape
        if (
            query is key is value
            and mask is not None
            and mask.shape == (B, 1, 1, L_q)
            and mask.dtype == torch.bool
            and USEFLASHATTEN
        ):
            # 仅支持 Encoder 自注意力的 padding mask
            Q = self.q_proj(query).reshape(B, L_q, self.num_heads, self.head_dim)
            K = self.k_proj(key).reshape(B, L_q, self.num_heads, self.head_dim)
            V = self.v_proj(value).reshape(B, L_q, self.num_heads, self.head_dim)
            qkv = torch.stack([Q, K, V], dim=2)  # (B, L, 3, H, head_dim)
            qkv = qkv.reshape(B * L_q, 3, self.num_heads, self.head_dim)  # (total_q, 3, H, head_dim)
            lengths = mask.squeeze(1).squeeze(1).sum(dim=1).to(torch.int32)  # (B,)
            assert lengths.dtype == torch.int32, "Lengths must be int32 for flash_attn_varlen_qkvpacked_func"
            out = flash_attn_varlen_qkvpacked_func(
                qkv, lengths, dropout_p=self.dropout.p if self.training else 0.0, causal=False,max_seqlen=512
            )
            out = out.reshape(B, L_q, D) # type: ignore
            return self.out_proj(out)
        else:
            Q = self.q_proj(query)  # (B, L_q, D)
            K = self.k_proj(key)    # (B, L_k, D)
            V = self.v_proj(value)

            Q = Q.reshape(B, L_q, self.num_heads, self.head_dim)
            K = K.reshape(B, -1, self.num_heads, self.head_dim)  # (B, L_k, H, head_dim)
            V = V.reshape(B, -1, self.num_heads, self.head_dim)  # (B, L_v, H, head_dim)

            Q = Q.transpose(1, 2)  # (B, H, L_q, head_dim)
            K = K.transpose(1, 2)  # (B, H, L_k, head_dim)
            V = V.transpose(1, 2)  # (B, H, L_v, head_dim)

            scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, L_q, L_k)
            scores = scores / math.sqrt(self.head_dim)  # 缩放
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))

            attn_weights = torch.softmax(scores, dim=-1)  # (B, H, L_q, L_k)
            attn_output = torch.matmul(attn_weights, V)  # (B, H, L_q, head_dim)
            attn_output = attn_output.transpose(1, 2)# (B, L_q, H, head_dim)
            attn_output = attn_output.reshape(B, L_q, self.d_model)  # (B, L_q, D)
            attn_output = self.dropout(attn_output)

            attn_output = self.out_proj(attn_output)  # (B, L_q, D)
            return attn_output  # 返回形状 (B, L_q, D)


# -----------------------------------------------------------------------------
# Position‑wise Feed‑Forward Network
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    """FFN with shape (D -> d_ff -> D)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# -----------------------------------------------------------------------------
# Encoder / Decoder layers
# -----------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """单层 Encoder 前向。

        Args:
            x (Tensor): ``(B, L_src, D)`` 输入序列。
            src_mask (Tensor): ``(B,1,1,L_src)`` 源序列 padding mask。
        Returns:
            Tensor: ``(B, L_src, D)`` 同形状输出。
        """
        # ------------------------------------------------------------------
        # TODO (学生实现)：按顺序实现 (1) 自注意力 + 残差 + LN → (2) FFN + 残差 + LN
        # ------------------------------------------------------------------
        # (1) 自注意力
        atten_out = self.self_attn(x, x, x, src_mask)  # (B, L_src, D)
        atten_out = self.dropout(atten_out)
        x = self.norm1(x + atten_out)  # 残差连接 + 层归一化 (B, L_src, D)
        # (2) 前馈网络
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)  # (B, L_src, D)
        x = self.norm2(x + ffn_out)  # 残差连接 + 层归一化 (B, L_src, D)
        return x  # 返回形状 (B, L_src, D)


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor, # Encoder 输出 (B, L_src, D) 也就是飞线残差
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """单层 Decoder 前向。

        Args:
            x (Tensor): ``(B, L_tgt, D)`` 目标序列嵌入。
            memory (Tensor): ``(B, L_src, D)`` Encoder 输出。
            tgt_mask (Tensor): ``(B,1,L_tgt,L_tgt)`` 目标序列掩码。
            src_mask (Tensor): ``(B,1,1,L_src)`` 源序列掩码。
        Returns:
            Tensor: ``(B, L_tgt, D)`` 同形状输出。
        """
        # ------------------------------------------------------------------
        # TODO (学生实现)：实现 (1) 掩码自注意力 → (2) 编解码注意力 → (3) FFN，
        # 每步均需加残差及层归一化。
        # ------------------------------------------------------------------
        # (1) 掩码自注意力
        atten_out = self.self_attn(x, x, x, tgt_mask)
        atten_out = self.dropout(atten_out)
        x = self.norm1(x + atten_out)  # 残差连接 + 层归一化 (B, L_tgt, D)
        # (2) 编解码注意力
        cross_out = self.cross_attn(x, memory, memory, src_mask)
        atten_out = self.dropout(cross_out)  # (B, L_tgt, D)
        x = self.norm2(x + cross_out)  # 残差连接 + 层归一化 (B, L_tgt, D)
        # (3) 前馈网络
        ffn_out = self.ffn(x)  #(B, L_tgt, D)
        ffn_out = self.dropout(ffn_out)
        x = self.norm3(x + ffn_out)  # 残差连接 + 层归一化 (B, L_tgt, D)
        return x  # 返回形状 (B, L_tgt, D)

# -----------------------------------------------------------------------------
# Encoder & Decoder stacks
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(src) #from (B, L_src, vocab_size) to (B, L_src, D)
        x = self.pe(x) # add positional encoding
        for layer in self.layers:
            assert isinstance(x, torch.Tensor), "Layer output must be a Tensor, but got {}".format(type(x))
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed(tgt)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


# -----------------------------------------------------------------------------
# Top‑level Seq2Seq model
# -----------------------------------------------------------------------------
class Seq2SeqTransformer(nn.Module):
    """Transformer wrapper exposing the API expected by train / eval scripts."""

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        d_model = emb_size

        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_encoder_layers,
            nhead,
            dim_feedforward,
            dropout,
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            num_decoder_layers,
            nhead,
            dim_feedforward,
            dropout,
        )
        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------
    def _make_src_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        # (B,1,1,L_src) – True for *valid* tokens
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def _make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        # Padding mask
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
        L = tgt.size(1)
        subsequent_mask = torch.triu(
            torch.ones(L, L, device=tgt.device, dtype=torch.bool), diagonal=1
        )  # (L,L)
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
        return pad_mask & ~subsequent_mask  # (B,1,L,L)

    # ------------------------------------------------------------------
    # Public API matching utils.translate_sentence
    # ------------------------------------------------------------------
    def encode(self, src_ids: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        if src_mask is None:
            src_mask = self._make_src_key_padding_mask(src_ids)
        return self.encoder(src_ids, src_mask)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ):
        if tgt_mask is None:
            tgt_mask = self._make_tgt_mask(tgt_ids)
        return self.decoder(tgt_ids, memory, tgt_mask, src_mask)

    def generator(self, dec_out: torch.Tensor) -> torch.Tensor:
        return self.proj(dec_out)

    # ------------------------------------------------------------------
    # Standard forward used in training
    # ------------------------------------------------------------------
    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        src_mask = self._make_src_key_padding_mask(src_ids)
        tgt_mask = self._make_tgt_mask(tgt_ids)

        memory = self.encoder(src_ids, src_mask)
        dec_out = self.decoder(tgt_ids, memory, tgt_mask, src_mask)
        return self.proj(dec_out)
