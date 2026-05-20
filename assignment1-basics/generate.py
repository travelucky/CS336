from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from cs336_basics.model import Tokenizer
from cs336_basics.tiny_lm import TinyTransformerConfig, TinyTransformerLM


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a tiny Assignment 1 Transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs" / "tinystories_tiny" / "checkpoint_final.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little girl")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str) -> tuple[TinyTransformerLM, Tokenizer]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TinyTransformerConfig(**checkpoint["config"])
    model = TinyTransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_data = checkpoint["tokenizer"]
    tokenizer = Tokenizer(
        tokenizer_data["vocab"],
        tokenizer_data["merges"],
        tokenizer_data["special_tokens"],
    )
    return model, tokenizer


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return indices.gather(-1, sampled)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: TinyTransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    end_token = tokenizer.special_token_to_id.get("<|endoftext|>")

    for _ in range(max_new_tokens):
        context = ids[:, -model.config.context_length :]
        logits = model(context)[:, -1, :]
        next_id = sample_next_token(logits, temperature=temperature, top_k=top_k)
        ids = torch.cat([ids, next_id], dim=1)
        if end_token is not None and next_id.item() == end_token:
            break

    return tokenizer.decode(ids.squeeze(0).tolist())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    model, tokenizer = load_model(args.checkpoint, device)
    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


if __name__ == "__main__":
    main()
