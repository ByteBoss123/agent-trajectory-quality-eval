# Agent Trajectory Quality Evaluation

An offline evaluation of tool-using customer-service agents. It covers 1,980 published conversations from the [τ-bench](https://github.com/sierra-research/tau-bench) benchmark (Sierra, MIT license): GPT-4o and Claude 3.5 Sonnet (new) working as retail and airline support agents, with 165 tasks and 4–8 attempts per task.

The agent runs are Sierra's. This repo adds the evaluation layer on top of them:

- a deterministic replay harness that re-scores every conversation from its tool calls
- an audit of the benchmark's scoring
- reliability (pass^k) and failure root-cause analysis
- a policy-compliance check on database writes
- a test of whether label-free trace signals can work as an online quality monitor

All numbers below come from `results/report.json`, which `python src/analyze.py` produces.

## Questions and findings

**1. Can the logged scores be reproduced?** Yes, with four exceptions. Replaying each conversation's tool calls against the benchmark's simulated databases reproduces 1,976 of 1,980 logged rewards. Three of the four mismatches are runs that crashed and were logged as 0. The fourth (Sonnet, airline task 11, trial 1) is unexplained.

**2. Were the published scores right?** No, they were inflated. When these runs were logged (Nov 2024), tasks that required a specific answer were scored on that answer alone, ignoring whether the database ended in the correct state. Sierra fixed this in Dec 2024 (PR #16). Re-scored with the fixed metric, 110 logged passes become failures, all on answer-required tasks:

| Agent / domain | Logged | Corrected (95% CI, tasks resampled) | pass^2 | pass^4 | pass^8 |
|---|---|---|---|---|---|
| Sonnet / retail | 69.2% | 61.0% (54.7–67.5) | 47.7% | 35.5% | 25.2% |
| GPT-4o / retail | 60.4% | 54.6% (47.2–61.7) | 42.3% | 30.4% | – |
| Sonnet / airline | 46.8% | 45.0% (35.8–54.5) | 31.3% | 21.4% | 16.0% |
| GPT-4o / airline | 42.5% | 42.5% (33.0–53.0) | 28.0% | 20.0% | – |

pass^k is the probability that an agent succeeds on all k attempts at the same task. Reliability falls steeply: Sonnet/retail drops from 61% on a single attempt to 25% across eight.

**3. How do agents fail?** Usually with wrong arguments to the right action. Each of the 903 failures was bucketed by comparing the agent's database writes to the benchmark's annotated actions:

| Bucket | Share |
|---|---|
| Right write tool, wrong arguments | 43.3% |
| Never made the needed change | 18.7% |
| Missed one of several needed writes | 13.0% |
| Wrong kind of write | 11.3% |
| Changed something no change was needed for | 8.2% |
| Correct database, missing required answer | 5.1% |
| Other | 0.4% |

The buckets come from rules written for this project. They are an approximation, not annotator labels.

**4. Do agents follow the confirmation policy?** Not always. Both domains' written policy requires listing the action details and getting an explicit "yes" before any database write. Across 2,762 writes, the share whose preceding user turn contained any form of assent (yes, sure, ok, go ahead, and similar) ranged from 79.6% (GPT-4o/airline) to 96.7% (Sonnet/retail). The share with an explicit yes-type word ranged from 67.6% to 94.0%.

The outcome metric doesn't capture this. **55 of 813 passing conversations that made writes (6.8%) included at least one write with no assent at all**, yet the task counted as a success. I reviewed a random sample of 40 flagged writes (`results/unconfirmed_write_audit_sample.csv`) myself; no human annotators were involved. None contained an explicit confirmation. About 5 were direct instructions ("Cancel both. Reason: no longer needed.") that a lenient reviewer might accept as consent. Retail's authenticate-first rule held: only 1 write happened before identity lookup.

**5. Can label-free trace signals detect failures online?** Only weakly, and mostly by spotting hard tasks rather than bad runs. A gradient-boosted classifier used 11 signals that are available without ground truth: conversation length, tool errors, writes, unconfirmed writes, transfers, user pushback, agent apologies, and user text volume. It reached:

- ROC-AUC 0.683 (95% CI 0.633–0.728), with folds grouped by task so no task appears in both training and testing. The base failure rate is 45.6%.
- 69.2% precision and 30.3% recall when flagging the riskiest 20% of conversations.

When task difficulty is held fixed (comparing failed and passed attempts *of the same task*, over 3,275 pairs), discrimination collapses to **0.524**, with logistic regression at 0.496. No single signal goes beyond 0.40–0.55 within a task. More writes within a task actually signals *success* (0.40), consistent with "never made the change" being a common failure. Cross-model transfer, training on one agent and testing on the other with disjoint tasks, gives 0.62 and 0.60, with lower CI bounds of 0.53 and 0.51.

**Takeaway:** surface trace statistics mostly tell you a task is hard, not that a given run went wrong. An online monitor needs semantic checks, such as action-argument verification or policy checks like #4, or delayed outcome labels.

## What was tried and dropped

A "policy guard" counterfactual replayed conversations with unconfirmed writes blocked. It appeared to break more runs than it fixed (16 fixed, 70 broken). That result reflects a limit of the replay simulator, not of the guard: replay can block a write, but it can't simulate the agent then asking for confirmation and the user saying yes. So blocking is not equivalent to enforcing the policy. Testing a guard properly needs live re-runs with a user simulator.

## Method notes

- **Replay** uses τ-bench's own tool implementations and data at commit `b581e61`, the commit that added these trajectories. A copy-on-access database view keeps replay fast. On 30 sampled trajectories it agrees exactly with τ-bench's full-database SHA-256 comparison, and a test checks this.
- **Confidence intervals** use a 2,000-draw bootstrap that resamples tasks, because repeated attempts at one task are correlated.
- **Confirmation detection** is regex-based (strict and lenient vocabularies), so treat it as a bound rather than ground truth.
- **Limitations:** these are 2024 agents, and Sierra's README now points to newer benchmark versions (τ²/τ³-bench). The simulated users are LLMs, not people.

## Run it

```bash
# from the folder that contains this repo
git clone https://github.com/sierra-research/tau-bench.git
git -C tau-bench worktree add ../tau-bench-hist b581e61
cp tau-bench-hist/historical_trajectories/*.json agent-trajectory-quality-eval/data/
cd agent-trajectory-quality-eval
pip install -r requirements.txt
python src/analyze.py        # ~2 min on one CPU; writes results/
python src/build_tableau.py  # writes dashboard/ CSVs + agent_quality.twb
pytest -q tests              # 9 tests
```

The trajectory files (~50 MB) aren't committed; the step above copies them in. To keep the benchmark elsewhere, set `TAU_BENCH_DIR`.

**Dashboard:** open `dashboard/agent_quality.twb` in Tableau Desktop or Tableau Public with the four CSVs in the same folder. Tableau may ask you to locate the CSVs on first open. The workbook validates against Tableau's 2026.2 XML schema but has not yet been opened in Tableau.
