"""Train one LoRA adapter per task family on a frozen Qwen2.5-VL base.

  python finetune_lora.py --data data/change.jsonl --out adapters/change \
      --base Qwen/Qwen2.5-VL-7B-Instruct --epochs 2

Only the adapter is trained (the base stays frozen), so the five specialists
share one set of base weights at serving time:

  vllm serve Qwen/Qwen2.5-VL-7B-Instruct --enable-lora --max-lora-rank 16 \
      --lora-modules satquery-vqa=adapters/vqa satquery-caption=adapters/caption \
                     satquery-grounding=adapters/grounding satquery-change=adapters/change \
                     satquery-fusion=adapters/fusion

and the backend (SATQUERY_VLM_PROVIDER=openai_compat) routes each task to its
adapter by name, per app/data/registry.yaml.

Catastrophic forgetting (PPT risk #2) is limited by the frozen base, a small
rank, and mixing `--replay-fraction` of general-domain examples back in.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments


class ChatJsonl(Dataset):
    def __init__(self, paths: list[Path], replay: Path | None, replay_fraction: float, seed: int):
        rows = [json.loads(l) for p in paths for l in p.read_text().splitlines() if l.strip()]
        if replay and replay_fraction > 0:
            extra = [json.loads(l) for l in replay.read_text().splitlines() if l.strip()]
            random.Random(seed).shuffle(extra)
            rows += extra[: int(len(rows) * replay_fraction)]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    """Tokenises chat + images and masks everything except the assistant turn,
    so the loss is only on the answer the adapter should learn to produce."""

    def __init__(self, processor, max_pixels: int):
        self.p = processor
        self.max_pixels = max_pixels

    def _messages(self, row, with_answer: bool):
        msgs = []
        for m in row["messages"]:
            if m["role"] == "assistant" and not with_answer:
                continue
            if m["role"] == "user":
                content = [{"type": "image"} for _ in row["images"]] + [{"type": "text", "text": m["content"]}]
            else:
                content = [{"type": "text", "text": m["content"]}]
            msgs.append({"role": m["role"], "content": content})
        return msgs

    def __call__(self, batch):
        images, full_texts, prompt_lens = [], [], []
        for row in batch:
            imgs = [Image.open(p).convert("RGB") for p in row["images"]]
            for im in imgs:
                im.thumbnail((int(self.max_pixels**0.5),) * 2)
            images.append(imgs)
            full = self.p.apply_chat_template(self._messages(row, True), tokenize=False)
            prompt = self.p.apply_chat_template(self._messages(row, False), tokenize=False, add_generation_prompt=True)
            full_texts.append(full)
            prompt_lens.append(len(self.p(text=[prompt], images=[imgs], return_tensors="pt")["input_ids"][0]))
        enc = self.p(text=full_texts, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, n in enumerate(prompt_lens):
            labels[i, :n] = -100
        enc["labels"] = labels
        return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=512 * 512)
    ap.add_argument("--replay", type=Path, help="general-domain chat JSONL mixed back in against forgetting")
    ap.add_argument("--replay-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.base, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            # language-model attention + MLP projections; the vision tower stays frozen
            target_modules=r".*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)",
        ),
    )
    model.print_trainable_parameters()

    ds = ChatJsonl(args.data, args.replay, args.replay_fraction, args.seed)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out), num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
            bf16=True, logging_steps=20, save_strategy="epoch", report_to=[], remove_unused_columns=False,
            lr_scheduler_type="cosine", warmup_ratio=0.03, seed=args.seed,
        ),
        train_dataset=ds,
        data_collator=Collator(processor, args.max_pixels),
    )
    trainer.train()
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)


if __name__ == "__main__":
    main()
