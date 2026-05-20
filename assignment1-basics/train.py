from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import (
    AdamW,
    Tokenizer,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    train_bpe,
)
from cs336_basics.tiny_lm import TinyTransformerConfig, TinyTransformerLM


PROJECT_ROOT = Path(__file__).resolve().parent
SPECIAL_TOKENS = ["<|endoftext|>"]


def default_input_path() -> Path: # 自动决定用哪个文本文件训练
    candidates = [
        PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
        PROJECT_ROOT / "tests" / "fixtures" / "tinystories_sample_5M.txt",
        PROJECT_ROOT / "tests" / "fixtures" / "tinystories_sample.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find TinyStories data. Put TinyStoriesV2-GPT4-train.txt under data/ "
        "or pass --input-path."
    )


def read_corpus(path: Path, max_chars: int | None) -> str: # 从文本文件里读字符串
    with path.open("r", encoding="utf-8") as f:
        if max_chars is None:
            return f.read()
        return f.read(max_chars)


def prepare_corpus_file(text: str, out_dir: Path) -> Path: # 把这次实际用来训练的文本保存一份
    corpus_path = out_dir / "tinystories_subset.txt"
    corpus_path.write_text(text, encoding="utf-8")
    return corpus_path


def build_tokenizer(corpus_path: Path, vocab_size: int, out_dir: Path) -> Tokenizer:
    vocab, merges = train_bpe(corpus_path, vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer_path = out_dir / "tokenizer.pt"
    torch.save(
        {"vocab": vocab, "merges": merges, "special_tokens": SPECIAL_TOKENS},
        tokenizer_path,
    )
    return Tokenizer(vocab, merges, SPECIAL_TOKENS)


def encode_corpus(tokenizer: Tokenizer, text: str) -> np.ndarray:
    token_ids = tokenizer.encode(text)
    return np.array(token_ids, dtype=np.int64)


def split_dataset(token_ids: np.ndarray, context_length: int) -> tuple[np.ndarray, np.ndarray]: # 把 token 序列切成训练集和验证集
    min_split = context_length + 1
    if len(token_ids) < 2 * min_split:
        raise ValueError(
            f"Need at least {2 * min_split} tokens for train/valid splits, got {len(token_ids)}. "
            "Use a larger corpus or a shorter --context-length."
        )
    val_count = max(min_split, min(len(token_ids) // 10, context_length * 200))
    train_ids = token_ids[:-val_count]
    valid_ids = token_ids[-val_count:]
    if len(train_ids) <= context_length or len(valid_ids) <= context_length:
        raise ValueError("Train/valid split is too small for the requested context length.")
    return train_ids, valid_ids


def compute_lm_loss(model: TinyTransformerLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: # 算语言模型 loss
    logits = model(x)
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


# 在验证集上算平均 loss
@torch.no_grad() 
def evaluate( 
    model: TinyTransformerLM,
    valid_ids: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(valid_ids, batch_size, context_length, device)
        losses.append(compute_lm_loss(model, x, y).item())
    model.train()
    return float(np.mean(losses))


def write_loss_csv(history: list[dict[str, float | int | None]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "valid_loss", "lr"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _svg_points(points: list[tuple[int, float]], x_min: int, x_max: int, y_min: float, y_max: float) -> str:
    left, top, width, height = 70, 40, 800, 380
    if x_max == x_min:
        x_max += 1
    if math.isclose(y_max, y_min):
        y_max += 1.0
    coords = []
    for step, loss in points:
        x = left + (step - x_min) / (x_max - x_min) * width
        y = top + (y_max - loss) / (y_max - y_min) * height
        coords.append(f"{x:.2f},{y:.2f}")
    return " ".join(coords)


def write_loss_svg(history: list[dict[str, float | int | None]], path: Path) -> None:
    train_points = [(int(row["step"]), float(row["train_loss"])) for row in history if row["train_loss"] is not None]
    valid_points = [(int(row["step"]), float(row["valid_loss"])) for row in history if row["valid_loss"] is not None]
    all_points = train_points + valid_points
    if not all_points:
        return

    x_min = min(step for step, _ in all_points)
    x_max = max(step for step, _ in all_points)
    y_min = min(loss for _, loss in all_points)
    y_max = max(loss for _, loss in all_points)
    padding = max((y_max - y_min) * 0.08, 0.1)
    y_min -= padding
    y_max += padding

    train_polyline = _svg_points(train_points, x_min, x_max, y_min, y_max)
    valid_polyline = _svg_points(valid_points, x_min, x_max, y_min, y_max)
    y_ticks = [y_min + (y_max - y_min) * i / 4 for i in range(5)]
    tick_markup = []
    for tick in y_ticks:
        y = 40 + (y_max - tick) / (y_max - y_min) * 380
        tick_markup.append(
            f'<line x1="64" y1="{y:.2f}" x2="870" y2="{y:.2f}" stroke="#e5e7eb" />'
            f'<text x="58" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#4b5563">{tick:.2f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="940" height="500" viewBox="0 0 940 500">
  <rect width="940" height="500" fill="#ffffff"/>
  <text x="70" y="26" font-size="20" font-family="Arial, sans-serif" fill="#111827">TinyStories training loss</text>
  {"".join(tick_markup)}
  <line x1="70" y1="420" x2="870" y2="420" stroke="#374151" />
  <line x1="70" y1="40" x2="70" y2="420" stroke="#374151" />
  <text x="470" y="470" font-size="14" font-family="Arial, sans-serif" text-anchor="middle" fill="#374151">step</text>
  <text x="20" y="230" font-size="14" font-family="Arial, sans-serif" text-anchor="middle" transform="rotate(-90 20 230)" fill="#374151">loss</text>
  <text x="70" y="445" font-size="12" font-family="Arial, sans-serif" text-anchor="middle" fill="#4b5563">{x_min}</text>
  <text x="870" y="445" font-size="12" font-family="Arial, sans-serif" text-anchor="middle" fill="#4b5563">{x_max}</text>
  <polyline points="{train_polyline}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="{valid_polyline}" fill="none" stroke="#f97316" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="735" cy="24" r="5" fill="#2563eb"/><text x="748" y="29" font-size="13" font-family="Arial, sans-serif" fill="#374151">train</text>
  <circle cx="795" cy="24" r="5" fill="#f97316"/><text x="808" y="29" font-size="13" font-family="Arial, sans-serif" fill="#374151">valid</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def save_training_state(
    model: TinyTransformerLM,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out_path: Path,
    tokenizer: Tokenizer,
    history: list[dict[str, float | int | None]],
    args: argparse.Namespace,
) -> None:
    train_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
            "config": model.config_dict,
            "tokenizer": {
                "vocab": tokenizer.vocab,
                "merges": tokenizer.merges,
                "special_tokens": tokenizer.special_tokens,
            },
            "history": history,
            "train_args": train_args,
        },
        out_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyStories tiny training run for Assignment 1 components.")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "runs" / "tinystories_tiny")
    parser.add_argument("--max-chars", type=int, default=1_000_000)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads.")
    if (args.d_model // args.num_heads) % 2 != 0:
        raise ValueError("RoPE needs an even head dimension; adjust --d-model or --num-heads.")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.input_path if args.input_path is not None else default_input_path()
    text = read_corpus(input_path, args.max_chars)
    corpus_path = prepare_corpus_file(text, args.out_dir)
    print(f"Using corpus: {input_path}")
    print(f"Training tokenizer on {len(text):,} characters...")
    tokenizer = build_tokenizer(corpus_path, args.vocab_size, args.out_dir)
    token_ids = encode_corpus(tokenizer, text)
    train_ids, valid_ids = split_dataset(token_ids, args.context_length)
    np.save(args.out_dir / "train_ids.npy", train_ids)
    np.save(args.out_dir / "valid_ids.npy", valid_ids)
    print(f"Tokenized to {len(token_ids):,} tokens ({len(train_ids):,} train, {len(valid_ids):,} valid).")

    config = TinyTransformerConfig(
        vocab_size=len(tokenizer.vocab),
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model = TinyTransformerLM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float | int | None]] = []

    print(f"Training on {device} for {args.steps} steps...")
    model.train()
    for step in range(1, args.steps + 1):
        lr = get_lr_cosine_schedule(step, args.lr, args.min_lr, args.warmup_steps, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_ids, args.batch_size, args.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_lm_loss(model, x, y)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        valid_loss: float | None = None
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            valid_loss = evaluate(
                model,
                valid_ids=valid_ids,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_batches=args.eval_batches,
            )

        history.append(
            {
                "step": step,
                "train_loss": float(loss.item()),
                "valid_loss": valid_loss,
                "lr": float(lr),
            }
        )

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            msg = f"step {step:4d}/{args.steps} | train {loss.item():.4f} | lr {lr:.2e}"
            if valid_loss is not None:
                msg += f" | valid {valid_loss:.4f}"
            print(msg)

        if step % args.save_every == 0:
            save_training_state(
                model,
                optimizer,
                step,
                args.out_dir / "checkpoint_latest.pt",
                tokenizer,
                history,
                args,
            )

    save_training_state(
        model,
        optimizer,
        args.steps,
        args.out_dir / "checkpoint_final.pt",
        tokenizer,
        history,
        args,
    )
    write_loss_csv(history, args.out_dir / "losses.csv")
    write_loss_svg(history, args.out_dir / "loss.svg")
    print(f"Saved checkpoint: {args.out_dir / 'checkpoint_final.pt'}")
    print(f"Saved loss curve: {args.out_dir / 'loss.svg'}")


if __name__ == "__main__":
    main()
