"""Build per-task-family training JSONL for the LoRA adapters.

Input: any image-text dataset exported to JSONL (BigEarthNet.txt, VRSBench,
RSVQA, CDVQA, LEVIR-CC ...), one record per line, with the field names given
on the command line -- this script deliberately does not hard-code any
dataset's schema. Records are mapped to the five SatQuery task families and
stratified-subsampled so no family or label dominates (the PPT's answer to
"BigEarthNet.txt has ~9.6M annotations").

Output: one JSONL per task family, in chat format:
  {"images": ["path1", ...], "messages": [{"role": "user", "content": "..."},
                                           {"role": "assistant", "content": "<ANSWER>...</ANSWER>"}]}

Example:
  python prepare_data.py --src vrsbench_vqa.jsonl --image-field image --question-field question \
      --answer-field ground_truth --task vqa --stratify-field type --per-stratum 4000 --out data/
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

TASKS = ("vqa", "caption", "grounding", "change", "fusion")

SYSTEM_HINT = (
    "You are a remote-sensing imagery analyst. Answer only from the image(s) and the evidence given; "
    "wrap every number in <CLAIM value=\"..\" unit=\"..\">..</CLAIM> inside <ANSWER>..</ANSWER>."
)


def _get(rec: dict, dotted: str):
    cur = rec
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--image-field", required=True, help="field with an image path, or a list of paths (pairs)")
    ap.add_argument("--question-field", required=True)
    ap.add_argument("--answer-field", required=True)
    ap.add_argument("--task", choices=TASKS, help="task family for every record")
    ap.add_argument("--task-field", help="per-record task family field (values must be in TASKS)")
    ap.add_argument("--stratify-field", help="field to balance on (question type, LULC label, ...)")
    ap.add_argument("--per-stratum", type=int, default=5000)
    ap.add_argument("--image-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.task or args.task_field):
        ap.error("give --task or --task-field")

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with args.src.open() as f:
        for line in f:
            rec = json.loads(line)
            task = args.task or _get(rec, args.task_field)
            if task not in TASKS:
                continue
            stratum = str(_get(rec, args.stratify_field)) if args.stratify_field else "all"
            buckets[(task, stratum)].append(rec)

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    writers = {t: (args.out / f"{t}.jsonl").open("w") for t in TASKS}
    counts: dict[str, int] = defaultdict(int)
    for (task, _), recs in sorted(buckets.items()):
        rng.shuffle(recs)
        for rec in recs[: args.per_stratum]:
            imgs = _get(rec, args.image_field)
            imgs = imgs if isinstance(imgs, list) else [imgs]
            answer = str(_get(rec, args.answer_field)).strip()
            if not answer.startswith("<ANSWER>"):
                answer = f"<ANSWER>{answer}</ANSWER>"
            writers[task].write(json.dumps({
                "images": [str(args.image_root / p) for p in imgs],
                "messages": [
                    {"role": "system", "content": SYSTEM_HINT},
                    {"role": "user", "content": str(_get(rec, args.question_field))},
                    {"role": "assistant", "content": answer},
                ],
            }) + "\n")
            counts[task] += 1
    for w in writers.values():
        w.close()
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
