"""
训练脚本
Usage:
    python train.py -c config.yaml
"""
import argparse
import importlib
import math
import os
from pathlib import Path
from typing import Any, Dict

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tokenizer import BaseTokenizer
from utils import load_dataset, collate_fn
from model.transformer import Seq2SeqTransformer
import torch.cuda.amp as amp

from torch.utils.tensorboard import SummaryWriter

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="yaml 配置文件路径",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def instantiate_tokenizer(cfg: Dict[str, Any]) -> BaseTokenizer:
    tok_path = cfg.get("tokenizer", "tokenizer.JiebaEnTokenizer")
    mod_name, cls_name = tok_path.rsplit(".", 1)
    TokCls: type[BaseTokenizer] = getattr(importlib.import_module(mod_name), cls_name)
    return TokCls()


# ---------------------------------------------------------------------------
# 训练 & 验证
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc="train", leave=False)
    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)
        tgt_inp = tgt[:, :-1]
        tgt_gold = tgt[:, 1:]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # 自动混合精度
            logits = model(src, tgt_inp)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_gold.reshape(-1),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    return epoch_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    val_loss = 0.0
    for src, tgt in tqdm(dataloader, desc="valid", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        tgt_inp = tgt[:, :-1]
        tgt_gold = tgt[:, 1:]

        with torch.cuda.amp.autocast():
            logits = model(src, tgt_inp)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_gold.reshape(-1),
            )
        val_loss += loss.item()
    return val_loss / len(dataloader)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    summary_writer = SummaryWriter(cfg.get("train", {}).get("log_dir", "runs"))

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 3407))

    # ---------------- Tokenizer & Dataset ----------------
    tokenizer = instantiate_tokenizer(cfg)
    train_ds, val_ds, _ = load_dataset(cfg, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_token_id),
    )

    # ---------------- Model ----------------
    model = Seq2SeqTransformer(
        num_encoder_layers=cfg["model"]["enc_layers"],
        num_decoder_layers=cfg["model"]["dec_layers"],
        emb_size=cfg["model"]["emb_size"],
        nhead=cfg["model"]["nhead"],
        src_vocab_size=tokenizer.src_vocab_size,
        tgt_vocab_size=tokenizer.tgt_vocab_size,
        dim_feedforward=cfg["model"]["ffn_dim"],
        dropout=cfg["model"].get("dropout", 0.1),
        pad_id=tokenizer.pad_token_id,
    ).to(device)
    
    #fp16
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"src vocab size: {tokenizer.src_vocab_size}, tgt vocab size: {tokenizer.tgt_vocab_size}")
    print(model)

    # ---------------- Optimizer / Scheduler / Loss ----------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    
    num_training_steps = len(train_loader) * cfg["train"]["epochs"]
    num_warmup_steps = int(num_training_steps * cfg["train"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # ---------------- 训练循环 ----------------
    best_val = math.inf
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        print(f"\n===== Epoch {epoch}/{cfg['train']['epochs']} =====")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device
        )
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
        )
        summary_writer.add_scalar("loss/train", train_loss, epoch)
        summary_writer.add_scalar("loss/valid", val_loss, epoch)

        # 保存验证集上的最佳模型（可选）
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = save_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokenizer_state": {
                        "src_vocab": tokenizer.src_vocab,
                        "tgt_vocab": tokenizer.tgt_vocab,
                    },
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"New best model saved to {ckpt_path}")
    summary_writer.close()
    print("Training finished!")


if __name__ == "__main__":
    main()
