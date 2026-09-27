"""Pluggable narration backends.

The deterministic tool layer never depends on which one is active --
swapping providers changes how the *sentences* are produced, never the
numbers, and every backend's output goes through the same adjudicator.

- mock           offline, template-driven from ledger evidence (default)
- openai_compat  a self-hosted open-weight VLM (e.g. Qwen2.5-VL served by
                 vLLM with `--lora-modules`); each task is routed to its
                 LoRA adapter by passing the adapter name as the model
- anthropic      Claude, via the official SDK
"""
from __future__ import annotations

import base64
import io
import random
from abc import ABC, abstractmethod

from PIL import Image

from app.config import settings
from app.models.schemas import InputConfig, SufficiencyVerdict, Target, Task

_NAMES = {"water": "water", "vegetation": "vegetation", "built_up": "built-up area", "bare_soil": "bare soil"}


class VLMClient(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, images: list[Image.Image], context: dict, adapter: str | None = None) -> str:
        """`context` carries the structured task/evidence the prompt was built
        from; real providers use `prompt` + `images`, the mock uses `context`."""


def _b64_png(img: Image.Image, max_side: int = 1024) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- mock


def _claim_pct(frac: float) -> str:
    p = frac * 100
    return f'<CLAIM value="{p:.1f}" unit="percent">{p:.1f}%</CLAIM>'


def _claim_ha(ha: float) -> str:
    return f'<CLAIM value="{ha:.1f}" unit="hectares">{ha:,.1f} ha</CLAIM>'


class MockVLMClient(VLMClient):
    """Deterministic narration built from the ledger evidence. Every CLAIM
    value is copied from the measurements, so it passes adjudication --
    unless `perturb` is set, in which case one figure is deliberately
    misstated to demonstrate that the adjudicator catches and corrects it."""

    name = "mock"

    def generate(self, prompt: str, images: list[Image.Image], context: dict, adapter: str | None = None) -> str:
        spec = context["spec"]
        m: dict[str, float] = context["measurements"]
        d: dict = context["decisions"]
        suff = context["sufficiency"]
        config: InputConfig = context["config"]

        if suff == SufficiencyVerdict.SCOPE_DOWN:
            text = self._scope_down(spec, m, d, config, context.get("sufficiency_reason", ""), context.get("missing_input"))
        elif spec.task == Task.CHANGE:
            text = self._change(m, d)
        elif spec.task == Task.FUSION:
            text = self._fusion(m, d)
        elif spec.task == Task.GROUNDING:
            text = self._grounding(spec, m, d, context.get("regions", []))
        elif spec.task == Task.CAPTION or spec.target == Target.LAND_COVER:
            text = self._caption(m, d, config)
        else:
            text = self._vqa(spec, m, d, config, context.get("regions", []))

        if context.get("perturb"):
            text = self._perturb(text)
        return f"<ANSWER>{text}</ANSWER>"

    @staticmethod
    def _perturb(text: str) -> str:
        import re

        def bump(match: re.Match) -> str:
            v = float(match.group(1)) * random.Random(7).choice([1.6, 1.8, 0.5])
            return f'<CLAIM value="{v:.1f}" unit="{match.group(2)}">{v:.1f}{"%" if match.group(2) == "percent" else " ha"}</CLAIM>'

        return re.sub(r'<CLAIM value="([\d.]+)" unit="(percent|hectares)">[^<]*</CLAIM>', bump, text, count=1)

    @staticmethod
    def _key(m: dict, base: str, config: InputConfig) -> str | None:
        suffixes = {InputConfig.CROSS_MODAL_PAIR: ["_optical", "_sar"], InputConfig.BI_TEMPORAL_PAIR: ["_t2", "_t1"]}.get(config, [""])
        for s in suffixes:
            if f"{base}{s}" in m:
                return f"{base}{s}"
        return base if base in m else None

    def _vqa(self, spec, m, d, config, regions) -> str:
        t = spec.target.value
        fk, hk = self._key(m, f"{t}_fraction", config), self._key(m, f"{t}_area_ha", config)
        if fk is None:
            return "The tools could not measure that directly from this imagery, so no quantitative answer is given."
        name = _NAMES[t]
        frac, ha = m[fk], m[hk]
        where = d.get("region_hint", "the scene")
        when = " in the later image" if fk.endswith("_t2") else ""
        sensor = " (optical estimate)" if fk.endswith("_optical") else (" (SAR estimate)" if fk.endswith("_sar") else "")
        if spec.metric == "presence":
            if frac < 0.005:
                return f"Essentially no {name} was detected{when}: it covers only {_claim_pct(frac)} of the scene."
            lead = f"Yes. {name.capitalize()} covers {_claim_pct(frac)} of the scene{when}{sensor}, about {_claim_ha(ha)}"
        elif spec.metric == "area":
            lead = f"{name.capitalize()} covers about {_claim_ha(ha)}{when}{sensor}, which is {_claim_pct(frac)} of the scene"
        else:
            lead = f"{name.capitalize()} covers {_claim_pct(frac)} of the scene{when}{sensor}, about {_claim_ha(ha)}"
        text = f"{lead}, concentrated in {where}."
        if "iou" in m and config == InputConfig.CROSS_MODAL_PAIR:
            text += f' The optical and SAR estimates overlap with an IoU of <CLAIM value="{m["iou"]:.2f}" unit="iou">{m["iou"]:.2f}</CLAIM>.'
        return text

    def _caption(self, m, d, config) -> str:
        parts = []
        for t in ("vegetation", "water", "built_up", "bare_soil"):
            k = self._key(m, f"{t}_fraction", config)
            if k is not None and m[k] >= 0.005:
                parts.append((m[k], t))
        if not parts:
            return "The band evidence for this scene was insufficient to break the land cover down into classes."
        parts.sort(reverse=True)
        clauses = [f"{_NAMES[t]} ({_claim_pct(f)})" for f, t in parts]
        dominant = _NAMES[parts[0][1]]
        listing = ", ".join(clauses[:-1]) + (f" and {clauses[-1]}" if len(clauses) > 1 else clauses[0])
        return f"The scene is dominated by {dominant}. Measured cover: {listing}. The largest {dominant} area lies in {d.get('region_hint', 'the scene')}."

    def _grounding(self, spec, m, d, regions) -> str:
        if not regions:
            return f"No contiguous {_NAMES.get(spec.target.value, 'target')} region large enough to localise was found."
        top = regions[0]
        label = top["label"].replace("_", " ")
        n = len(regions)
        loc = ""
        if top.get("centroid_lonlat"):
            lon, lat = top["centroid_lonlat"]
            loc = f", centred near {lat:.4f}°N, {lon:.4f}°E"
        return (
            f"{n} {label} region{'s' if n > 1 else ''} {'were' if n > 1 else 'was'} located. The largest covers {_claim_ha(top['area_ha'])} "
            f"in {d.get('region_hint', 'the scene')}{loc}; all regions are boxed on the overlay."
        )

    def _change(self, m, d) -> str:
        label = d.get("change_indicator")
        if label is None:
            return "No indicator could be measured on both dates, so change was not quantified."
        name = _NAMES[label]
        direction = d.get("direction", "no significant change")
        g, l, t1, t2 = m[f"{label}_gained_ha"], m[f"{label}_lost_ha"], m[f"{label}_t1_area_ha"], m[f"{label}_t2_area_ha"]
        where = d.get("region_hint", "the scene")
        if direction == "increase":
            return (
                f"{name.capitalize()} increased between the two dates: {_claim_ha(g)} is newly {name} and {_claim_ha(l)} was lost, "
                f"taking it from {_claim_ha(t1)} to {_claim_ha(t2)}. The new {name} is concentrated in {where}."
            )
        if direction == "decrease":
            return (
                f"{name.capitalize()} decreased between the two dates: {_claim_ha(l)} was lost and {_claim_ha(g)} gained, "
                f"taking it from {_claim_ha(t1)} to {_claim_ha(t2)}. The loss is concentrated in {where}."
            )
        return f"No significant change in {name} was measured: {_claim_ha(g)} gained against {_claim_ha(l)} lost."

    def _fusion(self, m, d) -> str:
        if "iou" not in m:
            return "Fusion evidence was unavailable: no phenomenon could be measured by both sensors."
        iou = m["iou"]
        t = d.get("agreement_target", "water")
        word = "closely" if iou > 0.7 else ("moderately" if iou > 0.4 else "poorly")
        of = m.get(f"{t}_fraction_optical")
        sf = m.get(f"{t}_fraction_sar")
        extra = ""
        if of is not None and sf is not None:
            extra = f" Optical measures {_claim_pct(of)} {_NAMES[t]} and SAR measures {_claim_pct(sf)}."
        trust = (
            "Because two sensors with different failure modes agree, the shared area is a high-confidence result."
            if iou > 0.7
            else "Areas seen by only one sensor are marked separately and should be treated with caution."
        )
        return f'The optical- and SAR-derived {_NAMES[t]} masks agree {word} (IoU <CLAIM value="{iou:.2f}" unit="iou">{iou:.2f}</CLAIM>).{extra} {trust}'

    def _scope_down(self, spec, m, d, config, reason, missing) -> str:
        t = spec.target.value if spec.target != Target.LAND_COVER else None
        fk = self._key(m, f"{t}_fraction", config) if t else None
        if fk is None:
            # Target itself not measurable: report what is.
            measured = [(m[k], k.split("_fraction")[0]) for k in m if k.endswith("_fraction") and k.split("_fraction")[0] in _NAMES]
            measured.sort(reverse=True)
            if not measured:
                return f"This question cannot be answered from the uploaded imagery. {missing or 'Additional imagery'} would be needed."
            listing = ", ".join(f"{_NAMES[c]} {_claim_pct(f)}" for f, c in measured[:3])
            return f"{_NAMES.get(t, 'that').capitalize()} cannot be measured from this input. What it does measure: {listing}. To answer the question, provide {missing}."
        return (
            f"{_NAMES[t].capitalize()} currently covers {_claim_pct(m[fk])} of this scene, about {_claim_ha(m[fk.replace('_fraction', '_area_ha')])}, "
            f"concentrated in {d.get('region_hint', 'the scene')}. Whether that is abnormal cannot be determined from a single "
            f"acquisition; {missing or 'a pre-event image'} would let the system measure the actual change."
        )


# ---------------------------------------------------------------- open-weight VLM with LoRA adapters


class OpenAICompatVLMClient(VLMClient):
    """Self-hosted open-weight VLM behind an OpenAI-compatible
    /v1/chat/completions endpoint (vLLM, SGLang, LM Studio, Ollama).

    vLLM serves LoRA adapters as named models:
        vllm serve Qwen/Qwen2.5-VL-7B-Instruct --enable-lora \\
            --lora-modules satquery-vqa=./adapters/vqa satquery-change=./adapters/change ...
    so routing a task to its specialist is just choosing the model name."""

    name = "openai_compat"

    def __init__(self):
        import httpx

        self._http = httpx.Client(base_url=settings.vlm_base_url.rstrip("/"), timeout=settings.vlm_timeout_s)
        self._headers = {"Authorization": f"Bearer {settings.vlm_api_key}"} if settings.vlm_api_key else {}

    def generate(self, prompt: str, images: list[Image.Image], context: dict, adapter: str | None = None) -> str:
        model = adapter if (adapter and settings.vlm_use_adapters) else settings.vlm_model
        content: list[dict] = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64_png(i)}"}} for i in images]
        content.append({"type": "text", "text": prompt})
        resp = self._http.post(
            "/v1/chat/completions",
            headers=self._headers,
            json={"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 700, "temperature": 0.2},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- Claude


class AnthropicVLMClient(VLMClient):
    name = "anthropic"

    def __init__(self):
        import anthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("SATQUERY_ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate(self, prompt: str, images: list[Image.Image], context: dict, adapter: str | None = None) -> str:
        content: list[dict] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64_png(img)}} for img in images
        ]
        content.append({"type": "text", "text": prompt})
        message = self._client.messages.create(
            model=settings.anthropic_model,
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        if message.stop_reason == "refusal":
            return "<ANSWER>The narration model declined to describe this scene; the measured evidence is shown below.</ANSWER>"
        return "".join(block.text for block in message.content if block.type == "text")


_client_singleton: VLMClient | None = None


def get_vlm_client() -> VLMClient:
    global _client_singleton
    if _client_singleton is None:
        if settings.vlm_provider == "anthropic":
            _client_singleton = AnthropicVLMClient()
        elif settings.vlm_provider == "openai_compat":
            _client_singleton = OpenAICompatVLMClient()
        else:
            _client_singleton = MockVLMClient()
    return _client_singleton


def get_mock_client() -> VLMClient:
    return MockVLMClient()
