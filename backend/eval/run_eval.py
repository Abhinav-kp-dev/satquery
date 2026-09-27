"""SatQuery AI evaluation harness.

    cd backend && python -m eval.run_eval [--large 8192]

Sections
  A. Routing + sufficiency: task / target / verdict accuracy on a labelled
     question set (eval/routing_benchmark.jsonl), every miss listed.
  B. Tool accuracy: every measurement on the synthetic demo scenes vs the
     generator's ground-truth label maps (fraction error, mask IoU, change ha).
  C. Adjudication: (1) false-correction rate when the narrator is honest,
     (2) catch rate when the narrator is forced to misstate a figure.
  D. Latency per query, and for a large scene read through overviews.

Caveats, printed into the report too: the routing set was written by the
same team that wrote the rules, so it measures coverage, not
generalisation; the synthetic scenes are clean, so tool accuracy on them
is an upper bound, not a claim about real Sentinel/Cartosat/RISAT imagery.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SATQUERY_STORAGE_DIR", tempfile.mkdtemp(prefix="satquery-eval-"))
os.environ["SATQUERY_VLM_PROVIDER"] = "mock"
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402

from app.config import SAMPLE_DIR  # noqa: E402
from app.core.controller import create_session, run_query  # noqa: E402
from app.core.parser import parse_question  # noqa: E402
from app.core.preflight import LEGAL_TASKS_BY_CONFIG  # noqa: E402
from app.core.sufficiency import check_sufficiency  # noqa: E402
from app.models.schemas import ImageManifest, InputConfig, InputManifest, Modality  # noqa: E402
from app.samples import generate as gen  # noqa: E402
from app.tools.sar_backscatter import compute_sar_backscatter  # noqa: E402
from app.tools.spectral_index import compute_spectral_indices, primary_for  # noqa: E402

INPUTS = {
    "optical12": (InputConfig.SINGLE_IMAGE, [("optical", 12)]),
    "optical4": (InputConfig.SINGLE_IMAGE, [("optical", 4)]),
    "rgb3": (InputConfig.SINGLE_IMAGE, [("optical", 3)]),
    "sar2": (InputConfig.SINGLE_IMAGE, [("sar", 2)]),
    "sar1": (InputConfig.SINGLE_IMAGE, [("sar", 1)]),
    "bitemporal": (InputConfig.BI_TEMPORAL_PAIR, [("optical", 12), ("optical", 12)]),
    "crossmodal": (InputConfig.CROSS_MODAL_PAIR, [("optical", 12), ("sar", 2)]),
}


def _manifest(kind: str) -> InputManifest:
    config, imgs = INPUTS[kind]
    images = [
        ImageManifest(filename=f"{i}.tif", modality=Modality(m), band_count=b, width=64, height=64, dtype="uint16")
        for i, (m, b) in enumerate(imgs)
    ]
    return InputManifest(config=config, images=images, legal_tasks=LEGAL_TASKS_BY_CONFIG[config])


def eval_routing() -> tuple[str, dict]:
    rows = [json.loads(line) for line in (Path(__file__).parent / "routing_benchmark.jsonl").read_text().splitlines() if line.strip()]
    n = len(rows)
    ok = {"task": 0, "target": 0, "verdict": 0, "all": 0}
    misses = []
    for r in rows:
        m = _manifest(r["input"])
        spec = parse_question(r["q"], m.config, m.legal_tasks)
        verdict, _, _ = check_sufficiency(spec, m)
        got = {"task": spec.task.value, "target": spec.target.value, "verdict": verdict.value}
        hits = {k: got[k] == r[k] for k in ("task", "target", "verdict")}
        for k, v in hits.items():
            ok[k] += v
        ok["all"] += all(hits.values())
        if not all(hits.values()):
            misses.append(f"| {r['q']} | {r['input']} | {r['task']}/{r['target']}/{r['verdict']} | {got['task']}/{got['target']}/{got['verdict']} |")
    metrics = {k: ok[k] / n for k in ok}
    md = [
        "## A. Routing and evidence sufficiency",
        f"{n} labelled questions across 7 input types (`eval/routing_benchmark.jsonl`).",
        "",
        "| metric | accuracy |",
        "|---|---|",
        f"| task | {metrics['task']:.1%} |",
        f"| target | {metrics['target']:.1%} |",
        f"| sufficiency verdict | {metrics['verdict']:.1%} |",
        f"| all three correct | {metrics['all']:.1%} |",
        "",
        "Caveat: this set was written alongside the rules, so it measures rule coverage rather than generalisation "
        "to unseen phrasing. Misses (expected -> got):",
        "",
        "| question | input | expected | got |",
        "|---|---|---|---|",
        *misses,
    ]
    return "\n".join(md), metrics


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def eval_tools() -> tuple[str, dict]:
    gen.generate(SAMPLE_DIR)
    labels, aux = gen.base_labels()
    scenes = {
        "s2_2024-03-15_pre.tif": labels,
        "s2_2024-07-08_post.tif": gen.flooded(labels, aux),
        "s1_2024-07-08_vvvh.tif": gen.flooded(labels, aux),
    }
    rows = ["| scene | class | truth % | measured % | abs error (pts) | mask IoU |", "|---|---|---|---|---|---|"]
    ious = []
    for fname, lab in scenes.items():
        with rasterio.open(SAMPLE_DIR / fname) as src:
            raw = src.read().astype("float32")
            names = list(src.descriptions)
        if "s1_" in fname:
            res = compute_sar_backscatter(raw, raw.shape[0], 10.0, names)
            outs = {"water": res["mask"], "built_up": res["built_up_mask"]}
        else:
            res = compute_spectral_indices(raw, raw.shape[0], 10.0, names)
            outs = {c: primary_for(res, c)[1]["mask"] for c in ("water", "vegetation", "built_up", "bare_soil")}
        for cls, mask in outs.items():
            truth = np.isin(lab, [c for c, n in gen.CLASS_NAMES.items() if n == cls])
            iou = _iou(mask, truth)
            ious.append(iou)
            rows.append(f"| {fname} | {cls} | {truth.mean() * 100:.2f} | {mask.mean() * 100:.2f} | {abs(mask.mean() - truth.mean()) * 100:.2f} | {iou:.3f} |")

    truth = json.loads((SAMPLE_DIR / "ground_truth.json").read_text())["changes"]
    change_rows = ["| pair | indicator | truth gained ha | measured gained ha | error |", "|---|---|---|---|---|"]
    samples = {s["sample_id"]: s for s in gen.SAMPLES}
    change_errs = []
    for sid, q in (("flood_pair", "How much did the water extent change?"), ("mining_pair", "Has any new excavation appeared?")):
        info = create_session([(str(SAMPLE_DIR / f), f) for f in samples[sid]["files"]], sid)
        t = run_query(info.session_id, q)
        call = next(c for c in t.tool_calls if c.tool == "delta_change")
        label = call.output["change_indicator"]
        got = call.output[f"{label}_gained_ha"]
        exp = truth[sid]["gained_ha"]
        err = abs(got - exp) / exp
        change_errs.append(err)
        shift = info.manifest.alignment.residual_shift_px if info.manifest.alignment else None
        change_rows.append(f"| {sid} | {label} | {exp:.1f} | {got:.1f} | {err:.1%} (residual shift estimated {shift}) |")

    md = [
        "## B. Tool accuracy on synthetic scenes",
        "Measurements vs the generator's label maps. Optical scenes are noise-light, so these are an upper bound; "
        "the SAR scene carries 4-look gamma speckle and is the more realistic row.",
        "",
        *rows,
        "",
        "Change detection (the mining pair's later scene was shifted by (2, -1) px on purpose):",
        "",
        *change_rows,
    ]
    return "\n".join(md), {"mean_iou": float(np.mean(ious)), "max_change_error": float(max(change_errs))}


def eval_adjudication() -> tuple[str, dict]:
    honest_claims = honest_corrections = 0
    perturbed = caught = delivered_ok = 0
    for s in gen.SAMPLES:
        info = create_session([(str(SAMPLE_DIR / f), f) for f in s["files"]], s["sample_id"])
        for q in s["suggested_questions"]:
            t = run_query(info.session_id, q)
            honest_claims += len(t.claims)
            honest_corrections += sum(c.corrected for c in t.claims)
            t = run_query(info.session_id, q, perturb=True)
            if t.answer_raw and 'CLAIM' in t.answer_raw and t.claims:
                bad = [c for c in t.claims if c.corrected]
                perturbed += 1
                caught += bool(bad)
                delivered_ok += all(f"{c.value}" not in t.answer for c in bad)
    metrics = {
        "false_correction_rate": honest_corrections / max(honest_claims, 1),
        "catch_rate": caught / max(perturbed, 1),
        "delivered_clean_rate": delivered_ok / max(perturbed, 1),
    }
    md = [
        "## C. Adjudication",
        "Every sample question run twice: once with the honest offline narrator, once with the narrator forced to "
        "misstate one figure (the UI's stress-test switch).",
        "",
        "| metric | value |",
        "|---|---|",
        f"| figures checked (honest runs) | {honest_claims} |",
        f"| false corrections on honest runs | {honest_corrections} ({metrics['false_correction_rate']:.1%}) |",
        f"| answers with an injected wrong figure | {perturbed} |",
        f"| injected errors caught | {caught} ({metrics['catch_rate']:.1%}) |",
        f"| delivered answers free of the wrong figure | {delivered_ok} ({metrics['delivered_clean_rate']:.1%}) |",
        "",
        "Caveat: the injected errors are large (x0.5 to x1.8). A narrator that is wrong by less than the 10% "
        "tolerance is, by design, accepted.",
    ]
    return "\n".join(md), metrics


def _write_large(path: Path, size: int) -> None:
    rng = np.random.default_rng(0)
    tile = 1024
    profile = dict(driver="GTiff", width=size, height=size, count=4, dtype="uint16", crs=gen.CRS,
                   transform=rasterio.transform.from_origin(*gen.ORIGIN, 10.0, 10.0), tiled=True,
                   blockxsize=512, blockysize=512, compress="deflate", BIGTIFF="IF_SAFER")
    veg = np.array([400, 800, 500, 4000], dtype="uint16")[:, None, None]
    water = np.array([700, 600, 400, 200], dtype="uint16")[:, None, None]
    with rasterio.open(path, "w", **profile) as dst:
        for y in range(0, size, tile):
            for x in range(0, size, tile):
                block = np.broadcast_to(veg, (4, tile, tile)).copy()
                if (x // tile + y // tile) % 3 == 0:
                    block[:, : tile // 2] = water
                block = (block * rng.uniform(0.97, 1.03, size=(1, tile, tile))).astype("uint16")
                dst.write(block, window=rasterio.windows.Window(x, y, tile, tile))
        dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)


def eval_latency(large: int) -> tuple[str, dict]:
    per_query = []
    for s in gen.SAMPLES:
        info = create_session([(str(SAMPLE_DIR / f), f) for f in s["files"]], s["sample_id"])
        for q in s["suggested_questions"]:
            t0 = time.perf_counter()
            run_query(info.session_id, q)
            per_query.append((time.perf_counter() - t0) * 1000)
    md = [
        "## D. Latency (CPU only, offline narrator)",
        "",
        "| workload | value |",
        "|---|---|",
        f"| queries on 512x512 demo sessions | median {statistics.median(per_query):.0f} ms, max {max(per_query):.0f} ms (n={len(per_query)}) |",
    ]
    metrics = {"median_query_ms": statistics.median(per_query)}
    if large:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "large.tif"
            t0 = time.perf_counter()
            _write_large(p, large)
            build_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            info = create_session([(str(p), "large.tif")])
            sess_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            t = run_query(info.session_id, "How many hectares of water are there?")
            q_s = time.perf_counter() - t0
            img = info.manifest.images[0]
            ha = next(c.output["water_area_ha"] for c in t.tool_calls if c.tool == "spectral_index")
            n_tiles = (large // 1024) ** 2
            water_tiles = sum(1 for y in range(large // 1024) for x in range(large // 1024) if (x + y) % 3 == 0)
            truth_ha = water_tiles * 512 * 1024 * 0.01
            md.append(
                f"| {large}x{large} 4-band scene ({large * large / 1e6:.0f} MP, internal overviews) | "
                f"session {sess_s:.2f} s + query {q_s:.2f} s; analysed at {img.analysis_width}px / {img.analysis_gsd_m:.0f} m GSD; "
                f"water {ha:,.0f} ha vs {truth_ha:,.0f} ha truth ({abs(ha - truth_ha) / truth_ha:.1%} error); file build {build_s:.0f} s |"
            )
            metrics.update(large_session_s=sess_s, large_query_s=q_s)
            del n_tiles
    return "\n".join(md), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", type=int, default=8192, help="side of the large-scene latency test (0 to skip)")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "RESULTS.md")
    args = ap.parse_args()

    sections, summary = [], {}
    for name, fn in (("routing", eval_routing), ("tools", eval_tools), ("adjudication", eval_adjudication)):
        md, m = fn()
        sections.append(md)
        summary[name] = m
        print(f"[{name}] {json.dumps(m)}")
    md, m = eval_latency(args.large)
    sections.append(md)
    summary["latency"] = m
    print(f"[latency] {json.dumps(m)}")

    header = (
        "# SatQuery AI - evaluation results\n\n"
        "Generated by `python -m eval.run_eval`. Re-run it to reproduce; numbers below are from the last run.\n"
    )
    args.out.write_text(header + "\n\n".join(sections) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
