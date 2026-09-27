# Domain adaptation: per-task LoRA adapters

Part A of the development plan: adapt an open-weight VLM (Qwen2.5-VL) to
remote sensing with one small LoRA adapter per task family, on a frozen base.

> **Status:** these scripts are written but were **not run** in this repository.
> Running them needs a GPU and the datasets. The app works without them: the
> offline narrator and the Claude backend need no adapters, and every number in an answer
> comes from the deterministic tools and adjudicator either way.

| family | suggested sources (see the PPT's references) |
|---|---|
| vqa | RSVQA, VRSBench VQA, BigEarthNet.txt VQA |
| caption | VRSBench captions, BigEarthNet.txt captions |
| grounding | VRSBench / DIOR-RSVG referring expressions |
| change | CDVQA, LEVIR-CC change captions |
| fusion | BigEarthNet.txt (co-registered Sentinel-1 + Sentinel-2) |

```bash
pip install -r requirements.txt
# 1. export each dataset to JSONL, then map + stratify-subsample it
python prepare_data.py --src rsvqa.jsonl --image-field img --question-field question \
    --answer-field answer --task vqa --stratify-field type --per-stratum 4000 --out data/
# 2. train one adapter per family
python finetune_lora.py --data data/vqa.jsonl --out adapters/vqa
# 3. serve all five on one base and point the backend at it
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --enable-lora --max-lora-rank 16 \
  --lora-modules satquery-vqa=adapters/vqa satquery-change=adapters/change ...
SATQUERY_VLM_PROVIDER=openai_compat SATQUERY_VLM_BASE_URL=http://localhost:8001 uvicorn app.main:app
```

Before and after training, run `backend/eval/run_eval.py` and compare the
adjudication section. A better-adapted narrator shows up there as fewer
corrections on honest runs. The tool-accuracy numbers don't change, because the
adapters never produce measurements.
