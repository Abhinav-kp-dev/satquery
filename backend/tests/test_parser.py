import pytest

from app.core.parser import parse_question
from app.core.preflight import LEGAL_TASKS_BY_CONFIG
from app.models.schemas import InputConfig as C, Target as T, Task


def parse(q, config=C.SINGLE_IMAGE, prev=None):
    return parse_question(q, config, LEGAL_TASKS_BY_CONFIG[config], prev, "prev" if prev else None)


@pytest.mark.parametrize(
    "q,config,task,target",
    [
        ("How much of this scene is water?", C.SINGLE_IMAGE, Task.VQA, T.WATER),
        ("Describe the land cover in this image.", C.SINGLE_IMAGE, Task.CAPTION, T.LAND_COVER),
        ("Highlight the water body", C.SINGLE_IMAGE, Task.GROUNDING, T.WATER),
        ("What changed between these two dates?", C.BI_TEMPORAL_PAIR, Task.CHANGE, T.LAND_COVER),
        ("Is this area flooded?", C.BI_TEMPORAL_PAIR, Task.CHANGE, T.WATER),  # baseline language on a pair is change
        ("Use both images to find water", C.CROSS_MODAL_PAIR, Task.FUSION, T.WATER),
        ("Has new mining appeared?", C.BI_TEMPORAL_PAIR, Task.CHANGE, T.BARE_SOIL),
        ("How much urban area is there?", C.SINGLE_IMAGE, Task.VQA, T.BUILT_UP),
    ],
)
def test_routing(q, config, task, target):
    spec = parse(q, config)
    assert (spec.task, spec.target) == (task, target)


def test_word_boundaries():
    assert parse("Any streets in the season photo?").target == T.LAND_COVER  # not 'tree' / 'sea'
    assert parse("What is the exchange rate of this area?").task != Task.CHANGE


def test_change_intent_recorded_even_when_input_cannot_support_it():
    spec = parse("What changed since 2019?", C.SINGLE_IMAGE)
    assert spec.task == Task.CHANGE
    assert "intent_not_supported_by_input" in spec.routing_layer


def test_follow_up_inherits_target_and_amount_goes_to_vqa():
    first = parse("Use both sensors to map the water", C.CROSS_MODAL_PAIR)
    follow = parse("and in hectares?", C.CROSS_MODAL_PAIR, prev=first)
    assert follow.follow_up and follow.target == T.WATER and follow.task == Task.VQA and follow.metric == "area"


def test_explicit_target_is_not_a_follow_up():
    first = parse("Use both sensors to map the water", C.CROSS_MODAL_PAIR)
    nxt = parse("How much built-up area is there?", C.CROSS_MODAL_PAIR, prev=first)
    assert not nxt.follow_up and nxt.target == T.BUILT_UP


def test_metrics():
    assert parse("How many hectares of water?").metric == "area"
    assert parse("What percentage is vegetation?").metric == "fraction"
    assert parse("Is there any water?").metric == "presence"
    assert parse("Is this area flooded?").implied_baseline
