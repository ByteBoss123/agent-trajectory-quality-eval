"""Write tidy CSVs for the dashboard and a Tableau workbook (.twb) that reads them.

Open dashboard/agent_quality.twb in Tableau Desktop or Tableau Public with the CSVs in
the same folder (Tableau may ask you to locate them on first open)."""
import json
import uuid
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "dashboard"
OUT.mkdir(exist_ok=True)
CELL = {"gpt-4o": "GPT-4o", "sonnet-35-new": "Claude 3.5 Sonnet (new)"}


def label(m, d):
    return f"{CELL[m]} / {d}"


def tidy():
    pr = pd.read_csv(RES / "pass_rates.csv")
    pd.DataFrame([dict(Cell=label(r.model, r.domain), Metric=met, Rate=r[col])
                  for _, r in pr.iterrows()
                  for met, col in [("Logged (legacy metric)", "legacy"),
                                   ("Corrected (DB state + outputs)", "corrected")]]
                 ).to_csv(OUT / "success_rates.csv", index=False)
    pk = pd.read_csv(RES / "pass_k_long.csv")
    pd.DataFrame(dict(Cell=[label(m, d) for m, d in zip(pk.model, pk.domain)],
                      K=[f"k={k}" for k in pk.k], PassK=pk.pass_k)).to_csv(
        OUT / "pass_k.csv", index=False)
    ft = pd.read_csv(RES / "failure_taxonomy.csv")
    pd.DataFrame(dict(Cell=[label(m, d) for m, d in zip(ft.model, ft.domain)],
                      FailureType=ft.failure_type, Failures=ft.n)).to_csv(
        OUT / "failure_types.csv", index=False)
    sg = pd.read_csv(RES / "signals.csv")
    rep = json.load(open(RES / "report.json"))
    rows = [dict(Signal=r.signal, View=v, AUC=r[c]) for _, r in sg.iterrows()
            for v, c in [("Pooled (across tasks)", "pooled_auc"),
                         ("Within task (difficulty held fixed)", "within_task_auc")]]
    g = rep["detector_task_grouped_cv"]["gbm"]
    rows += [dict(Signal="GBM detector (all signals)", View="Pooled (across tasks)", AUC=g["auc"]),
             dict(Signal="GBM detector (all signals)", View="Within task (difficulty held fixed)",
                  AUC=g["within_task_auc"])]
    pd.DataFrame(rows).to_csv(OUT / "signal_auc.csv", index=False)


def ds_xml(name, fname, cols):
    """cols: [(col_name, datatype)] with datatype in {string, real, integer}."""
    conn = f"textscan.{uuid.uuid4().hex[:28]}"
    colxml = "\n".join(
        f"            <column datatype='{t}' name={quoteattr(c)} ordinal='{i}' />"
        for i, (c, t) in enumerate(cols))
    meta = "\n".join(
        f"""          <column datatype='{t}' name='[{c}]' role='{"measure" if t != "string" else "dimension"}' type='{"quantitative" if t != "string" else "nominal"}' />"""
        for c, t in cols)
    table = fname.replace(".", "#")
    return f"""    <datasource caption={quoteattr(name)} inline='true' name='federated.{name.lower()}' version='18.1'>
      <connection class='federated'>
        <named-connections>
          <named-connection caption={quoteattr(name)} name='{conn}'>
            <connection class='textscan' directory='.' filename='{fname}' password='' server='' />
          </named-connection>
        </named-connections>
        <relation connection='{conn}' name='{fname}' table='[{table}]' type='table'>
          <columns character-set='UTF-8' header='yes' locale='en_US' separator=','>
{colxml}
          </columns>
        </relation>
      </connection>
{meta}
    </datasource>"""


def ws_xml(title, ds, rows, cols, color, mark):
    """rows/cols/color: lists of (field, kind) with kind 'dim' or 'measure'."""
    def inst(f, k):
        return (f"[none:{f}:nk]" if k == "dim" else f"[sum:{f}:qk]")
    used = {(f, k) for f, k in rows + cols + color}
    deps = "\n".join(
        f"""            <column-instance column='[{f}]' derivation='{"None" if k == "dim" else "Sum"}' name='{inst(f, k)}' pivot='key' type='{"nominal" if k == "dim" else "quantitative"}' />"""
        for f, k in sorted(used))
    shelf = lambda xs: " / ".join(f"[federated.{ds}].{inst(f, k)}" for f, k in xs)  # noqa: E731
    enc = "".join(f"<color column='[federated.{ds}].{inst(f, k)}' />" for f, k in color)
    return f"""    <worksheet name={quoteattr(title)}>
      <table>
        <view>
          <datasources>
            <datasource name='federated.{ds}' />
          </datasources>
          <datasource-dependencies datasource='federated.{ds}'>
{deps}
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <style />
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view>
              <breakdown value='auto' />
            </view>
            <mark class='{mark}' />
            <encodings>{enc}</encodings>
          </pane>
        </panes>
        <rows>{shelf(rows)}</rows>
        <cols>{shelf(cols)}</cols>
      </table>
      <simple-id uuid='{{{str(uuid.uuid4()).upper()}}}' />
    </worksheet>"""


SHEETS = [
    ("Success rate: logged vs corrected", "success_rates",
     [("Cell", "dim")], [("Metric", "dim"), ("Rate", "measure")], [("Metric", "dim")], "Bar"),
    ("Reliability: pass^k", "pass_k",
     [("PassK", "measure")], [("K", "dim")], [("Cell", "dim")], "Line"),
    ("Failure root causes", "failure_types",
     [("FailureType", "dim")], [("Failures", "measure")], [("Cell", "dim")], "Bar"),
    ("Failure signals: pooled vs within-task AUC", "signal_auc",
     [("Signal", "dim")], [("View", "dim"), ("AUC", "measure")], [("View", "dim")], "Bar"),
]
SCHEMAS = {
    "success_rates": [("Cell", "string"), ("Metric", "string"), ("Rate", "real")],
    "pass_k": [("Cell", "string"), ("K", "string"), ("PassK", "real")],
    "failure_types": [("Cell", "string"), ("FailureType", "string"), ("Failures", "integer")],
    "signal_auc": [("Signal", "string"), ("View", "string"), ("AUC", "real")],
}


CARDS = """      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='31'>
            <card type='columns' />
          </strip>
          <strip size='31'>
            <card type='rows' />
          </strip>
          <strip size='31'>
            <card type='title' />
          </strip>
        </edge>
      </cards>"""


def workbook():
    dss = "\n".join(ds_xml(n, f"{n}.csv", c) for n, c in SCHEMAS.items())
    wss = "\n".join(ws_xml(*s) for s in SHEETS)
    W, H = 1400, 900
    zones, zid = [], 3
    for i, s in enumerate(SHEETS):
        x, y = (i % 2) * (W // 2), 40 + (i // 2) * ((H - 40) // 2)
        zones.append(f"          <zone h='{(H - 40) // 2}' id='{zid + i + 1}' name={quoteattr(s[0])} "
                     f"type-v2='visual' w='{W // 2}' x='{x}' y='{y}' />")
    dash = f"""    <dashboard name='Agent Quality Overview'>
      <size maxheight='{H}' maxwidth='{W}' minheight='{H}' minwidth='{W}' sizing-mode='fixed' />
      <datasources />
      <zones>
        <zone h='{H}' id='2' layout-strategy-id='flow' type-v2='layout-basic' w='{W}' x='0' y='0'>
          <zone h='40' id='3' type-v2='title' w='{W}' x='0' y='0' />
{chr(10).join(zones)}
        </zone>
      </zones>
      <simple-id uuid='{{{str(uuid.uuid4()).upper()}}}' />
    </dashboard>"""
    wins = "\n".join(
        f"    <window class='worksheet' name={quoteattr(s[0])}>\n{CARDS}\n"
        f"      <simple-id uuid='{{{str(uuid.uuid4()).upper()}}}' />\n    </window>" for s in SHEETS)
    vps = "".join(f"<viewpoint name={quoteattr(s[0])} />" for s in SHEETS)
    wins += (f"\n    <window class='dashboard' maximized='true' name='Agent Quality Overview'>\n"
             f"      <viewpoints>{vps}</viewpoints>\n      <active id='-1' />\n"
             f"      <simple-id uuid='{{{str(uuid.uuid4()).upper()}}}' />\n    </window>")
    xml = f"""<?xml version='1.0' encoding='utf-8' ?>
<!-- build 20262.26.0528.2205 -->
<workbook original-version='18.1' source-build='2026.2.0 (20262.26.0528.2205)' source-platform='mac' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <document-format-change-manifest>
    <SheetIdentifierTracking />
    <SortTagCleanup />
    <WindowsPersistSimpleIdentifiers />
  </document-format-change-manifest>
  <preferences />
  <datasources>
{dss}
  </datasources>
  <worksheets>
{wss}
  </worksheets>
  <dashboards>
{dash}
  </dashboards>
  <windows>
{wins}
  </windows>
</workbook>
"""
    (OUT / "agent_quality.twb").write_text(xml)


if __name__ == "__main__":
    tidy()
    workbook()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
