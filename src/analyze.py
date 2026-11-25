"""Agent-quality evaluation over 1,980 published tau-bench trajectories.

1. Outcome audit: logged vs replayed reward; legacy vs corrected metric; task-clustered CIs
2. Reliability: pass^k (same task attempted k times, all must succeed)
3. Failure taxonomy (offline; uses ground-truth actions)
4. Policy compliance: are DB writes preceded by explicit user confirmation?
5. Online failure detector from trace-only signals: task-grouped CV, within-task
   discrimination (controls for task difficulty), cross-model transfer
"""
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from replay import load_all, replay, tool_calls
from signals import RETAIL_AUTH, WRITE_TOOLS, confirmation, failure_type, trace_signals

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
RNG = np.random.default_rng(7)
B = 2000
FEATS = ["n_messages", "n_user_turns", "n_tool_calls", "n_tool_errors", "n_writes",
         "n_unconfirmed_writes", "transferred", "n_user_pushback", "n_agent_apology",
         "user_chars", "airline"]


def cluster_boot(df, col, group="task_id"):
    """Mean with 95% CI from resampling tasks (trials of one task are correlated)."""
    g = df.groupby(group)[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = RNG.integers(0, len(g), (B, len(g)))
    return s.sum() / c.sum(), *np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])


def pass_hat_k(df, k):
    vals = [comb(int(g.corrected.sum()), k) / comb(len(g), k)
            for _, g in df.groupby("task_id") if len(g) >= k]
    return float(np.mean(vals))


def auc_ci(y, p, groups):
    y, p, groups = map(np.asarray, (y, p, groups))
    uniq = np.unique(groups)
    pos = {gid: np.where(groups == gid)[0] for gid in uniq}
    out = []
    for _ in range(B):
        take = np.concatenate([pos[g] for g in RNG.choice(uniq, len(uniq))])
        if len(set(y[take])) == 2:
            out.append(roc_auc_score(y[take], p[take]))
    return float(roc_auc_score(y, p)), *map(float, np.percentile(out, [2.5, 97.5]))


def within_task_auc(y, s, groups):
    """P(score of a failed trial > score of a passed trial of the SAME task), ties = 0.5.
    Task difficulty is held fixed, so this isolates what the signal says about the run."""
    num = den = 0.0
    df = pd.DataFrame(dict(y=y, s=s, g=groups))
    for _, t in df.groupby("g"):
        f, p = t.s[t.y == 1].to_numpy(), t.s[t.y == 0].to_numpy()
        if len(f) and len(p):
            d = f[:, None] - p[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum()
            den += d.size
    return num / den if den else float("nan"), int(den)


def main():
    rows = load_all()
    recs, writes = [], []
    for r in rows:
        cor, ra, ro = replay(r["domain"], r["traj"], r["task"])
        leg, _, _ = replay(r["domain"], r["traj"], r["task"], metric="legacy")
        tid = f'{r["domain"]}-{r["task_id"]}'
        rec = dict(model=r["model"], domain=r["domain"], task_id=tid, trial=r["trial"],
                   logged=r["reward"], legacy=leg, corrected=cor, db_correct=int(ra),
                   outputs_correct=int(ro), has_outputs=bool(r["task"]["outputs"]),
                   run_error=r["run_error"])
        rec["failure_type"] = "pass" if cor == 1 else failure_type(
            r["domain"], r["traj"], r["task"], ra, ro)
        rec.update(trace_signals(r["domain"], r["traj"]))
        if r["domain"] == "retail":  # retail policy: authenticate via lookup before any write
            seen, early = False, 0
            for _, n, _ in tool_calls(r["traj"]):
                seen |= n in RETAIL_AUTH
                early += int(n in WRITE_TOOLS["retail"] and not seen)
            rec["writes_before_auth"] = early
        recs.append(rec)
        for c in confirmation(r["domain"], r["traj"]):
            writes.append(dict(model=r["model"], domain=r["domain"], task_id=tid,
                               trial=r["trial"], corrected=cor, **c))
    df, wf = pd.DataFrame(recs), pd.DataFrame(writes)
    df.to_csv(OUT / "trajectories.csv", index=False)
    wf.to_csv(OUT / "writes.csv", index=False)
    rep = {"n_trajectories": len(df), "n_tasks": int(df.task_id.nunique())}

    # 1. outcome audit
    mism = df[df.logged != df.legacy]
    rep["legacy_reproduction"] = dict(match=int(len(df) - len(mism)), of=len(df),
                                      mismatches_that_were_run_errors=int(mism.run_error.sum()))
    fp = df[(df.legacy == 1) & (df.corrected == 0)]
    rep["legacy_false_passes"] = dict(
        n=len(fp), all_on_required_output_tasks=bool(fp.has_outputs.all()),
        by_cell={f"{m}/{d}": int(n) for (m, d), n in fp.groupby(["model", "domain"]).size().items()})
    cells = []
    for (m, d), g in df.groupby(["model", "domain"]):
        lm, ll, lh = cluster_boot(g, "legacy")
        cm, cl, ch = cluster_boot(g, "corrected")
        row = dict(model=m, domain=d, trajectories=len(g), tasks=g.task_id.nunique(),
                   trials_per_task=int(g.groupby("task_id").size().min()),
                   legacy=lm, legacy_lo=ll, legacy_hi=lh, corrected=cm,
                   corrected_lo=cl, corrected_hi=ch)
        for k in (1, 2, 4, 8):
            if row["trials_per_task"] >= k:
                row[f"pass^{k}"] = pass_hat_k(g, k)
        cells.append(row)
    cells = pd.DataFrame(cells)
    cells.to_csv(OUT / "pass_rates.csv", index=False)
    rep["pass_rates"] = json.loads(cells.round(4).to_json(orient="records"))
    # long-format pass^k for the dashboard
    pk = [dict(model=r.model, domain=r.domain, k=k, pass_k=r[f"pass^{k}"])
          for _, r in cells.iterrows() for k in (1, 2, 4, 8) if pd.notna(r.get(f"pass^{k}"))]
    pd.DataFrame(pk).to_csv(OUT / "pass_k_long.csv", index=False)

    # 2-3. failure taxonomy
    fails = df[df.corrected == 0]
    rep["failure_taxonomy_share"] = fails.failure_type.value_counts(normalize=True).round(4).to_dict()
    (fails.groupby(["model", "domain"]).failure_type.value_counts().rename("n").reset_index()
     .to_csv(OUT / "failure_taxonomy.csv", index=False))

    # 4. policy compliance
    comp = {}
    for (m, d), g in wf.groupby(["model", "domain"]):
        comp[f"{m}/{d}"] = dict(writes=len(g), strict_confirmed=round(g.strict.mean(), 4),
                                lenient_confirmed=round(g.lenient.mean(), 4))
    passed_w = wf[wf.corrected == 1]
    rep["policy_compliance"] = dict(
        by_cell=comp,
        writes_total=len(wf),
        in_passing_trajectories=dict(writes=len(passed_w),
                                     unconfirmed_lenient=int((~passed_w.lenient).sum()),
                                     unconfirmed_strict=int((~passed_w.strict).sum())),
        passing_trajectories_with_any_unconfirmed_write=int(
            df[(df.corrected == 1) & (df.n_unconfirmed_writes > 0)].shape[0]),
        passing_trajectories_with_writes=int(df[(df.corrected == 1) & (df.n_writes > 0)].shape[0]),
        retail_writes_before_auth=int(df.writes_before_auth.fillna(0).sum()))
    audit = wf[~wf.lenient].sample(min(40, int((~wf.lenient).sum())), random_state=3)
    audit[["model", "domain", "task_id", "trial", "tool", "user_turn"]].to_csv(
        OUT / "unconfirmed_write_audit_sample.csv", index=False)

    # 5. failure detector
    X = df[FEATS].to_numpy(float)
    y = (df.corrected == 0).astype(int).to_numpy()
    grp = df.task_id.to_numpy()
    mk = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
          "gbm": lambda: GradientBoostingClassifier(random_state=0, max_depth=2,
                                                    n_estimators=150, learning_rate=0.05)}
    det, oof = {}, {}
    for name, f in mk.items():
        p = np.zeros(len(df))
        for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
            p[te] = f().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        a, lo, hi = auc_ci(y, p, grp)
        wa, pairs = within_task_auc(y, p, grp)
        det[name] = dict(auc=a, lo=lo, hi=hi, pr_auc=float(average_precision_score(y, p)),
                         within_task_auc=wa)
        oof[name] = p
    rep["failure_base_rate"] = float(y.mean())
    rep["detector_task_grouped_cv"] = det
    rep["within_task_pairs"] = pairs
    single = {}
    for f in FEATS[:-1]:
        single[f] = dict(pooled_auc=round(roc_auc_score(y, df[f]), 4),
                         within_task_auc=round(within_task_auc(y, df[f].to_numpy(), grp)[0], 4))
    rep["single_signal"] = single
    pd.DataFrame([dict(signal=k, **v) for k, v in single.items()]).to_csv(
        OUT / "signals.csv", index=False)

    tasks = np.array(sorted(df.task_id.unique()))
    RNG.shuffle(tasks)
    half = set(tasks[: len(tasks) // 2])
    xfer = {}
    for src, dst in [("gpt-4o", "sonnet-35-new"), ("sonnet-35-new", "gpt-4o")]:
        tr = ((df.model == src) & df.task_id.isin(half)).to_numpy()
        te = ((df.model == dst) & ~df.task_id.isin(half)).to_numpy()
        p = mk["logistic"]().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        a, lo, hi = auc_ci(y[te], p, grp[te])
        xfer[f"{src}->{dst}"] = dict(auc=a, lo=lo, hi=hi, n_test=int(te.sum()))
    rep["cross_model_transfer_disjoint_tasks"] = xfer

    best = max(det, key=lambda k: det[k]["auc"])
    p = oof[best]
    flag = p >= np.quantile(p, 0.8)
    rep["alert_top20pct"] = dict(model=best, precision=float(y[flag].mean()),
                                 recall=float(y[flag].sum() / y.sum()), base_rate=float(y.mean()))
    df.assign(risk_score=p).to_csv(OUT / "trajectories.csv", index=False)

    json.dump(rep, open(OUT / "report.json", "w"), indent=2, default=float)
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
