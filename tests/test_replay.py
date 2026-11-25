import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from replay import _BASE, full_hash, gt_state, load_all, replay, run_actions, same_state, tool_calls  # noqa: E402
from signals import LENIENT, STRICT, failure_type  # noqa: E402

ROWS = load_all()


def test_dataset_shape():
    assert len(ROWS) == 1980
    assert {(r["model"], r["domain"]) for r in ROWS} == {
        ("gpt-4o", "airline"), ("gpt-4o", "retail"),
        ("sonnet-35-new", "airline"), ("sonnet-35-new", "retail")}


def test_fast_state_compare_matches_official_hash():
    random.seed(0)
    for r in random.sample(ROWS, 30):
        a = run_actions(r["domain"], [(n, x) for _, n, x in tool_calls(r["traj"])])
        g = gt_state(r["domain"], r["task"])
        official = full_hash(r["domain"], a) == full_hash(r["domain"], g)
        assert official == same_state(r["domain"], a, g)


def test_replay_does_not_mutate_base_db():
    before = full_hash("retail", {t: {} for t in _BASE["retail"]})
    r = next(x for x in ROWS if x["domain"] == "retail" and x["reward"] == 1)
    replay(r["domain"], r["traj"], r["task"])
    assert full_hash("retail", {t: {} for t in _BASE["retail"]}) == before


@pytest.mark.parametrize("domain", ["retail", "airline"])
def test_legacy_metric_reproduces_logged_rewards(domain):
    sub = [r for r in ROWS if r["domain"] == domain and not r["run_error"]][:150]
    for r in sub:
        assert replay(r["domain"], r["traj"], r["task"], metric="legacy")[0] == r["reward"]


def test_corrected_metric_never_passes_what_legacy_fails():
    for r in ROWS[:300]:
        leg = replay(r["domain"], r["traj"], r["task"], metric="legacy")[0]
        cor = replay(r["domain"], r["traj"], r["task"])[0]
        assert cor <= leg


def test_skipping_every_call_matches_noop_outcome():
    r = next(x for x in ROWS if x["domain"] == "retail" and x["task"]["actions"]
             and any(a["name"].startswith(("cancel", "modify", "exchange", "return"))
                     for a in x["task"]["actions"]))
    n = len(tool_calls(r["traj"]))
    assert replay(r["domain"], r["traj"], r["task"], skip=frozenset(range(n)))[1] is False


def test_confirmation_lexicons():
    assert STRICT.search("Yes, please proceed") and LENIENT.search("Yes, please proceed")
    assert not STRICT.search("Fine, do it.") and LENIENT.search("Fine, do it.")
    assert not LENIENT.search("Weather-related - there's a storm warning for those dates.")


def test_failure_type_missing_info():
    task = {"actions": [], "outputs": ["42"]}
    assert failure_type("retail", [], task, True, False) == "missing_required_info"
