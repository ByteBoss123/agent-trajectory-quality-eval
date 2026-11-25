"""Two views of each trajectory.

trace_signals(): features observable in production WITHOUT ground truth, i.e. what an
  online quality monitor could log per conversation.
failure_type(): offline root-cause bucket that DOES use the annotated ground-truth
  actions. Used only for diagnosis, never as a detector feature.
confirmation(): policy-compliance check for database writes ("obtain explicit user
  confirmation (yes) before consequential actions", in both domains' written policy).
"""
import json
import re

from replay import tool_calls

WRITE_TOOLS = {
    "retail": {"cancel_pending_order", "exchange_delivered_order_items",
               "modify_pending_order_address", "modify_pending_order_items",
               "modify_pending_order_payment", "modify_user_address",
               "return_delivered_order_items"},
    "airline": {"book_reservation", "cancel_reservation", "update_reservation_baggages",
                "update_reservation_flights", "update_reservation_passengers",
                "send_certificate"},
}
RETAIL_AUTH = {"find_user_id_by_email", "find_user_id_by_name_zip"}

# Strict: the policy literally asks for an explicit "yes".
STRICT = re.compile(r"\b(yes|yeah|yep|yup|confirm(ed|s)?|i confirm|oui|s[ií])\b", re.I)
# Lenient: any assent phrasing a reasonable reviewer would accept.
LENIENT = re.compile(
    r"\b(yes|yeah|yep|yup|ya|sure|ok(ay)?|fine|confirm\w*|oui|s[ií]|correct|right|"
    r"go ahead|proceed\w*|please do|do it|do that|sounds good|that works|works for me|"
    r"let'?s (do|go|proceed)|go for it|perfect|great|absolutely|definitely|of course)\b",
    re.I)
PUSHBACK = re.compile(r"\b(not what|that's not|that is not|wrong|i said|i meant|incorrect|"
                      r"mistake)\b", re.I)
APOLOGY = re.compile(r"\b(apolog\w*|sorry|unfortunately|unable to)\b", re.I)


def last_user_before(traj, idx):
    for j in range(idx - 1, -1, -1):
        if traj[j]["role"] == "user":
            return traj[j].get("content") or ""
    return ""


def write_calls(domain, traj):
    """[(call_position, msg_index, name, args)] for database-changing calls."""
    return [(k, i, n, a) for k, (i, n, a) in enumerate(tool_calls(traj))
            if n in WRITE_TOOLS[domain]]


def confirmation(domain, traj):
    """Per write: was the most recent user turn an explicit (strict) / any (lenient) assent?"""
    out = []
    for _, i, n, _ in write_calls(domain, traj):
        u = last_user_before(traj, i)
        out.append(dict(tool=n, strict=bool(STRICT.search(u)), lenient=bool(LENIENT.search(u)),
                        user_turn=u))
    return out


def trace_signals(domain, traj):
    calls = tool_calls(traj)
    users = [m.get("content") or "" for m in traj if m["role"] == "user"]
    tools_out = [m.get("content") or "" for m in traj if m["role"] == "tool"]
    assistant = [m.get("content") or "" for m in traj if m["role"] == "assistant"]
    conf = confirmation(domain, traj)
    return {
        "n_messages": len(traj),
        "n_user_turns": len(users),
        "n_tool_calls": len(calls),
        "n_tool_errors": sum(t.strip().lower().startswith("error") for t in tools_out),
        "n_writes": len(conf),
        "n_unconfirmed_writes": sum(not c["lenient"] for c in conf),
        "transferred": int(any(n == "transfer_to_human_agents" for _, n, _ in calls)),
        "n_user_pushback": sum(bool(PUSHBACK.search(u)) for u in users[1:]),
        "n_agent_apology": sum(bool(APOLOGY.search(a)) for a in assistant),
        "user_chars": sum(len(u) for u in users),
        "airline": int(domain == "airline"),
    }


def _key(name, args):
    return name, json.dumps(args, sort_keys=True)


def failure_type(domain, traj, task, r_actions, r_outputs):
    """Root-cause bucket for a failed trajectory (ground truth used here only)."""
    if r_actions and not r_outputs:
        return "missing_required_info"
    gt = [_key(a["name"], a["kwargs"]) for a in task["actions"] if a["name"] in WRITE_TOOLS[domain]]
    ag = [_key(n, a) for _, _, n, a in write_calls(domain, traj)]
    gt_names, ag_names = {g[0] for g in gt}, {a[0] for a in ag}
    if gt and not ag:
        return "no_write"
    if not gt and ag:
        return "unneeded_write"
    if gt_names & ag_names and any(a not in gt for a in ag if a[0] in gt_names):
        return "wrong_arguments"
    if ag_names - gt_names:
        return "wrong_write_type"
    if set(gt) - set(ag):
        return "missed_write"
    return "other"
