"""Deterministic replay of logged tau-bench trajectories against the benchmark's own
simulated databases. Re-derives each trajectory's reward from its tool calls, so every
downstream metric rests on a reproduced outcome rather than only the logged label."""
import copy
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAU = Path(os.environ.get("TAU_BENCH_DIR", ROOT.parent / "tau-bench-hist"))
sys.path.insert(0, str(TAU))
sys.modules.setdefault("litellm", types.SimpleNamespace(completion=None))  # user simulator unused

from tau_bench.envs.base import consistent_hash, to_hashable  # noqa: E402
from tau_bench.envs.retail.tools import ALL_TOOLS as RETAIL_TOOLS  # noqa: E402
from tau_bench.envs.airline.tools import ALL_TOOLS as AIRLINE_TOOLS  # noqa: E402
from tau_bench.envs.retail.data import load_data as load_retail  # noqa: E402
from tau_bench.envs.airline.data import load_data as load_airline  # noqa: E402

DOMAINS = {
    "retail": ({t.get_info()["function"]["name"]: t for t in RETAIL_TOOLS}, load_retail),
    "airline": ({t.get_info()["function"]["name"]: t for t in AIRLINE_TOOLS}, load_airline),
}
_BASE = {d: fn() for d, (_, fn) in DOMAINS.items()}
TERMINATE = {"transfer_to_human_agents"}
DATA_DIR = ROOT / "data"


def tool_calls(traj):
    """[(message_index, name, args)] for every agent tool call, in order."""
    out = []
    for i, m in enumerate(traj):
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                out.append((i, tc["function"]["name"], args))
    return out


def responses(traj):
    return [m["content"] or "" for m in traj
            if m["role"] == "assistant" and not m.get("tool_calls")]


class CowTable(dict):
    """Copy-on-access view of a base table: a record is deep-copied the first time a tool
    touches it, so one replay costs O(records touched) instead of copying the whole DB."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def _get(self, k):
        if not dict.__contains__(self, k):
            dict.__setitem__(self, k, copy.deepcopy(self.base[k]))
        return dict.__getitem__(self, k)

    def __getitem__(self, k):
        return self._get(k)

    def get(self, k, default=None):
        return self._get(k) if k in self else default

    def __contains__(self, k):
        return dict.__contains__(self, k) or k in self.base

    def keys(self):
        return list(dict.fromkeys(list(self.base.keys()) + list(dict.keys(self))))

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return [(k, self._get(k)) for k in self.keys()]

    def values(self):
        return [self._get(k) for k in self.keys()]

    def __len__(self):
        return len(self.keys())


def run_actions(domain, actions):
    """Apply (name, args) actions; return {table: {key: record}} for records touched."""
    tools, _ = DOMAINS[domain]
    data = {t: CowTable(v) for t, v in _BASE[domain].items()}
    for name, args in actions:
        if name in TERMINATE or name not in tools:
            continue
        try:  # mirrors Env.step: a failing call returns an error string, DB unchanged
            tools[name].invoke(data=data, **args)
        except Exception:
            pass
    return {t: dict(dict.items(v)) for t, v in data.items()}


def same_state(domain, a, b):
    """Equivalent to comparing tau-bench's consistent_hash of the two full DB states (they
    share a base, so only records either side touched can differ)."""
    base = _BASE[domain]
    for t in base:
        for k in set(a[t]) | set(b[t]):
            ra, rb = a[t].get(k, base[t].get(k)), b[t].get(k, base[t].get(k))
            if str(to_hashable(ra)) != str(to_hashable(rb)):
                return False
    return True


def full_hash(domain, touched):
    """Official tau-bench hash of the full DB (slow; used only to validate same_state)."""
    data = {t: {**v, **touched[t]} for t, v in _BASE[domain].items()}
    return consistent_hash(to_hashable(data))


_GT = {}


def gt_state(domain, task):
    key = (domain, json.dumps(task["actions"], sort_keys=True))
    if key not in _GT:
        _GT[key] = run_actions(domain, [(a["name"], a["kwargs"]) for a in task["actions"]])
    return _GT[key]


def replay(domain, traj, task, skip=frozenset(), metric="corrected"):
    """Re-execute the agent's calls (minus call positions in `skip`) and score.

    metric="legacy": reward code when these trajectories were logged (Nov 2024): tasks with
      required outputs were scored on outputs only, ignoring final DB state.
    metric="corrected": current tau-bench reward (PR #16, Dec 2024): DB state AND outputs."""
    acts = [(n, a) for k, (_, n, a) in enumerate(tool_calls(traj)) if k not in skip]
    r_actions = same_state(domain, run_actions(domain, acts), gt_state(domain, task))
    text = [r.lower().replace(",", "") for r in responses(traj)]
    r_outputs = all(any(o.lower() in t for t in text) for o in task["outputs"])
    if metric == "legacy" and task["outputs"]:
        return float(r_outputs), r_actions, r_outputs
    return float(r_actions and r_outputs), r_actions, r_outputs


def load_all(data_dir=DATA_DIR):
    rows = []
    for f in sorted(Path(data_dir).glob("*.json")):
        model, domain = f.stem.rsplit("-", 1)
        for t in json.load(open(f)):
            rows.append(dict(model=model, domain=domain, task_id=t["task_id"],
                             trial=t["trial"], reward=t["reward"], traj=t["traj"],
                             task=t["info"]["task"],
                             run_error=t["info"].get("reward_info") is None))
    return rows
