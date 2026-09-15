"""Pluggable narration backend.

The deterministic tool layer never depends on which of these is active --
swapping providers only changes how the *sentences* are produced, never the
numbers. `mock` needs no credentials and is the default so the prototype
runs fully offline; `anthropic` sends the rendered composite(s) plus the
same prompt to Claude for genuinely model-generated narration.
"""
from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod

from PIL import Image

from app.config import settings
from app.models.schemas import SufficiencyVerdict, Task


class VLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, images: list[Image.Image], context: dict) -> str:
        """context carries the same task/evidence/sufficiency the prompt text
        was built from, as structured data -- real providers use `prompt` and
        `images`; the offline mock uses `context` directly instead of
        re-parsing its own prompt string."""
        ...


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}"


def _fmt_ha(x: float) -> str:
    return f"{x:.1f}"


class MockVLMClient(VLMClient):
    """Deterministic, template-driven narration built directly from the
    tool evidence. Every CLAIM value below is copied verbatim from the
    evidence dict passed in, so it will always pass adjudication -- this
    client exists to make the full pipeline runnable and demoable with
    zero external dependencies, not to simulate a fine-tuned model's
    fluency."""

    def generate(self, prompt: str, images: list[Image.Image], context: dict) -> str:
        task = context["task"]
        ev = context["evidence"]
        sufficiency = context["sufficiency"]
        region_hint = ev.get("region_hint", "the scene")

        if sufficiency == SufficiencyVerdict.SCOPE_DOWN:
            return self._scope_down(ev)

        if task == Task.CHANGE:
            return self._change(ev)
        if task == Task.FUSION:
            return self._fusion(ev)
        if task == Task.GROUNDING:
            return self._grounding(ev, region_hint)
        if task == Task.CAPTION:
            return self._caption(ev)
        return self._vqa(ev)

    def _pick_primary(self, ev: dict) -> tuple[str, float, str] | None:
        for key, unit in (
            ("water_fraction", "percent"),
            ("built_up_fraction", "percent"),
            ("vegetation_fraction", "percent"),
        ):
            if key in ev:
                return key, ev[key] * 100, unit
        return None

    def _vqa(self, ev: dict) -> str:
        primary = self._pick_primary(ev)
        if not primary:
            return "<ANSWER>The tool evidence available for this scene does not support a confident quantitative answer.</ANSWER>"
        key, pct, unit = primary
        label = key.replace("_fraction", "").replace("_", " ")
        return (
            f'<ANSWER>{label.capitalize()} covers <CLAIM value="{pct:.1f}" unit="{unit}">'
            f"approximately {pct:.1f}%</CLAIM> of this scene, measured directly from the imagery.</ANSWER>"
        )

    def _caption(self, ev: dict) -> str:
        clauses = []
        if "water_fraction" in ev:
            pct = ev["water_fraction"] * 100
            clauses.append(f'water covering <CLAIM value="{pct:.1f}" unit="percent">{pct:.1f}%</CLAIM>')
        if "vegetation_fraction" in ev:
            pct = ev["vegetation_fraction"] * 100
            clauses.append(f'vegetation covering <CLAIM value="{pct:.1f}" unit="percent">{pct:.1f}%</CLAIM>')
        if "built_up_fraction" in ev:
            pct = ev["built_up_fraction"] * 100
            clauses.append(f'built-up surfaces covering <CLAIM value="{pct:.1f}" unit="percent">{pct:.1f}%</CLAIM>')
        if not clauses:
            return "<ANSWER>This scene shows a mixed land-cover pattern; band evidence was insufficient to break it down further.</ANSWER>"
        return "<ANSWER>This scene shows a mixed land-cover pattern, with " + ", ".join(clauses) + " of the total area.</ANSWER>"

    def _grounding(self, ev: dict, region_hint: str) -> str:
        primary = self._pick_primary(ev)
        label = primary[0].replace("_fraction", "").replace("_", " ") if primary else "target region"
        return (
            f"<ANSWER>The {label} referred to in the question is concentrated in {region_hint}; "
            f"it has been marked on the overlay.</ANSWER>"
        )

    def _change(self, ev: dict) -> str:
        direction = ev.get("direction", "no significant change")
        gained = next((v for k, v in ev.items() if k.endswith("_gained_ha")), None)
        lost = next((v for k, v in ev.items() if k.endswith("_lost_ha")), None)
        if direction == "increase" and gained is not None:
            return (
                f'<ANSWER>The measured extent increased between the two dates, with <CLAIM value="{_fmt_ha(gained)}" '
                f'unit="hectares">about {_fmt_ha(gained)} hectares</CLAIM> newly present that was absent at the earlier date.</ANSWER>'
            )
        if direction == "decrease" and lost is not None:
            return (
                f'<ANSWER>The measured extent decreased between the two dates, with <CLAIM value="{_fmt_ha(lost)}" '
                f'unit="hectares">about {_fmt_ha(lost)} hectares</CLAIM> lost compared to the earlier date.</ANSWER>'
            )
        return "<ANSWER>No significant change was measured between the two dates for this indicator.</ANSWER>"

    def _fusion(self, ev: dict) -> str:
        iou = ev.get("iou")
        if iou is None:
            return "<ANSWER>Fusion evidence was unavailable for this scene.</ANSWER>"
        agreement_word = "closely" if iou > 0.7 else ("moderately" if iou > 0.4 else "poorly")
        return (
            f'<ANSWER>The optical-derived and SAR-derived masks agree {agreement_word}, with an overlap of '
            f'<CLAIM value="{iou:.2f}" unit="iou">{iou:.2f} IoU</CLAIM>. Regions where the two sensors disagree '
            f"are shown separately on the overlay and should be treated with lower confidence.</ANSWER>"
        )

    def _scope_down(self, ev: dict) -> str:
        primary = self._pick_primary(ev)
        if not primary:
            return (
                "<ANSWER>This question implies a comparison this single image cannot support on its own. "
                "A second, prior-date image of the same location would let the system measure the actual change.</ANSWER>"
            )
        key, pct, unit = primary
        label = key.replace("_fraction", "").replace("_", " ")
        return (
            f'<ANSWER>{label.capitalize()} covers <CLAIM value="{pct:.1f}" unit="{unit}">approximately {pct:.1f}%</CLAIM> '
            f"of this scene. Determining whether this reflects a genuine change or anomaly requires a baseline "
            f"image, which was not provided -- a second, prior-date image of this location would let the system "
            f"report the actual change rather than a single-point measurement.</ANSWER>"
        )


class AnthropicVLMClient(VLMClient):
    """Sends the rendered composite(s) plus the constructed prompt to
    Claude for narration. Used when SATQUERY_VLM_PROVIDER=anthropic and an
    API key is configured."""

    def __init__(self):
        import anthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("SATQUERY_ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    @staticmethod
    def _to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def generate(self, prompt: str, images: list[Image.Image], context: dict) -> str:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": self._to_b64(img)},
                }
            )
        message = self._client.messages.create(
            model=settings.anthropic_model,
            max_tokens=800,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(block.text for block in message.content if block.type == "text")


_client_singleton: VLMClient | None = None


def get_vlm_client() -> VLMClient:
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton
    if settings.vlm_provider == "anthropic":
        _client_singleton = AnthropicVLMClient()
    else:
        _client_singleton = MockVLMClient()
    return _client_singleton
