"""
Recompute entity-level NER metrics for the German-NER benchmark from the raw
per-sample predictions logged to Weights & Biases.

For every *finished* eval run in the W&B project it:
  1. reads the model / dataset / eval-strategy from the run,
  2. downloads the per-sample gold+prediction (the `samples_by_task` artifact,
     i.e. `<task>_eval_samples.json`),
  3. recomputes two *corpus-level (micro)* metrics from scratch:
       - micro_f1   : exact-string micro F1 over (label_type, entity) pairs.
       - partial_f1 : fuzzy micro F1 using token_set_ratio one-to-one matching
                      with a similarity threshold (lenient / "partial match").
  4. de-duplicates cells (dataset x model x strategy) by keeping the most
     recent run, and prints one large table plus a long-form CSV.

Why micro and not the logged `*_all_labels` numbers: the logged score averages
per-label F1 per sentence and awards 1.0 to empty-gold labels, which inflates
the result. Micro pooling counts only real entities, so absent labels no longer
inflate, and predicting entities where there are none is penalised as FP.

Run with the project venv:
    source /vol/auto_llm/venv/bin/activate
    python auto_llm/tracker/utils/german_ner_metrics_recompute.py
"""

import ast
import importlib.util
import json
import os
from collections import defaultdict

import pandas as pd
import wandb
from thefuzz import fuzz

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ENTITY = "llm4kmu"
PROJECT = "german-ner-eval"
# Europarl's clean re-run (train/eval on data/test) lives in a separate project;
# read it too so Europarl uses the NEW split instead of the contaminated 799.
NEWSPLIT_PROJECT = "german-ner-eval-newsplit"

# Fuzzy ("partial match") matching config.
FUZZY_SCORER = fuzz.token_set_ratio      # subset-tolerant, article/boundary robust
FUZZY_THRESHOLD = 0.90                    # similarity in [0,1] required to count a match

# Where to cache downloaded artifacts (kept out of the repo).
CACHE_ROOT = os.environ.get(
    "NER_RECOMPUTE_CACHE",
    "/tmp/claude-27699/-homes-mbrinner-auto-llm/ner_recompute_cache",
)
# Where to write the long-form CSV and the human/LaTeX tables text file.
CSV_OUT = os.environ.get("NER_RECOMPUTE_CSV", os.path.join(CACHE_ROOT, "recomputed_metrics.csv"))
TXT_OUT = os.environ.get("NER_RECOMPUTE_TXT", os.path.join(CACHE_ROOT, "ner_tables.txt"))

# The four models reported in the paper: pretrained-suffix -> display label.
TARGET_MODELS = {
    "Qwen3.5-9B": "Qwen 3.5 9B",
    "gemma-4-12B-it": "Gemma 4 12B",
    "Ministral-3-8B-Instruct-2512-BF16": "Ministral 8B",
    "gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
}
MODEL_ORDER = ["Qwen 3.5 9B", "Gemma 4 12B", "Ministral 8B", "GPT-5.4 mini"]

# task name (as logged) -> paper dataset display name. Defines the 12 datasets.
DATASET_MAP = {
    "fairagro_de_ner": "FAIRagro",
    "archaeo_ner": "Archaeo NER",
    "cofun_ner": "CO-Fun",
    "wiki_biography_ner": "WikiBiography",
    # NEW clean split; the original 'europarl_ner' (799, test carved into train)
    # is intentionally NOT mapped so its contaminated runs are ignored.
    "europarl_ner_newsplit": "Europarl-ner",
    "swiss_ner": "SwissNER",
    "nostad_ner": "NoSta-D",
    "ehri_holocaust_ner": "EHRI-NER",
    "zurich_state_archive_ner": "NER Staatsarchiv",
    "german_ler_ner": "German LER",
    "ger_ps_ner": "GerPS",
    "ger_med_ner": "GERNERMED",
    "humadex_german_ner": "HUMADEX German NER",
}
DATASET_ORDER = [
    "FAIRagro", "Archaeo NER", "CO-Fun", "WikiBiography", "Europarl-ner", "SwissNER", "NoSta-D",
    "EHRI-NER", "NER Staatsarchiv", "German LER", "GerPS", "GERNERMED", "HUMADEX German NER",
]

STRATEGY_ORDER = ["zero-shot", "few-shot", "lora"]

# A run whose predictions are empty/unparseable above this fraction is treated as a
# degenerate training collapse (immediate-EOS) and excluded from run selection.
DEGENERATE_FAIL_RATE = 0.5

# LaTeX body column order: dataset, then Ministral z/f/l, Qwen z/f/l, Gemma z/f/l, GPT z.
LATEX_COLS = [
    ("Ministral 8B", "zero-shot"), ("Ministral 8B", "few-shot"), ("Ministral 8B", "lora"),
    ("Qwen 3.5 9B", "zero-shot"), ("Qwen 3.5 9B", "few-shot"), ("Qwen 3.5 9B", "lora"),
    ("Gemma 4 12B", "zero-shot"), ("Gemma 4 12B", "few-shot"), ("Gemma 4 12B", "lora"),
    ("GPT-5.4 mini", "zero-shot"),
]

# --------------------------------------------------------------------------- #
# Reuse the repo's exact prediction-extraction logic so parsing matches the
# original evaluation byte-for-byte.
# --------------------------------------------------------------------------- #
_REPO_UTILS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "tasks", "sequence_to_structured_output", "ner", "utils.py",
)
_spec = importlib.util.spec_from_file_location("ner_task_utils", os.path.abspath(_REPO_UTILS))
_ner_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ner_utils)
clean_and_extract_json = _ner_utils.clean_and_extract_json
parse_dict = _ner_utils.parse_dict


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def get_gold(record):
    """Return the gold entity dict {label: [strings]} for a sample record."""
    doc = record.get("doc") or {}
    out = doc.get("output_text")
    if isinstance(out, dict):
        return out
    target = record.get("target")
    if isinstance(target, dict):
        return target
    if isinstance(target, str):
        try:
            return ast.literal_eval(target)
        except (ValueError, SyntaxError):
            try:
                return json.loads(target)
            except json.JSONDecodeError:
                return None
    return None


def get_pred(record):
    """Return the predicted entity dict, parsed exactly like the eval did.

    Returns None when the model output is unparseable (treated downstream as an
    all-empty prediction => every gold entity becomes a false negative)."""
    resp = None
    fr = record.get("filtered_resps")
    if isinstance(fr, list) and fr:
        resp = fr[0]
    if resp is None:
        rs = record.get("resps")
        if isinstance(rs, list) and rs and isinstance(rs[0], list) and rs[0]:
            resp = rs[0][0]
    if not isinstance(resp, str):
        return None
    return parse_dict(clean_and_extract_json(resp))


def as_str_list(value):
    """Coerce a label's value into a list of strings, mirroring the eval."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip() != ""]
    if value in (None, ""):
        return []
    return [str(value).strip()]


# --------------------------------------------------------------------------- #
# Metric counting (per (sample, label_type), then pooled = micro)
# --------------------------------------------------------------------------- #
def exact_counts(gold_list, pred_list):
    gs, ps = set(gold_list), set(pred_list)
    tp = len(gs & ps)
    return tp, len(ps - gs), len(gs - ps)          # tp, fp, fn


def fuzzy_counts(gold_list, pred_list, threshold, scorer):
    """One-to-one greedy matching above `threshold`; each entity used once."""
    pairs = []
    for gi, g in enumerate(gold_list):
        for pi, p in enumerate(pred_list):
            s = scorer(g, p) / 100.0
            if s >= threshold:
                pairs.append((s, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p = set(), set()
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
    tp = len(used_g)
    return tp, len(pred_list) - tp, len(gold_list) - tp    # tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_records(records):
    """Pool tp/fp/fn over all samples & label types -> micro exact & fuzzy F1."""
    e_tp = e_fp = e_fn = 0
    f_tp = f_fp = f_fn = 0
    n_parse_fail = 0
    for rec in records:
        gold = get_gold(rec)
        if not isinstance(gold, dict):
            continue
        pred = get_pred(rec)
        if not isinstance(pred, dict):
            n_parse_fail += 1
            pred = {}
        for label in set(gold) | set(pred):
            g = as_str_list(gold.get(label, []))
            p = as_str_list(pred.get(label, []))
            tp, fp, fn = exact_counts(g, p)
            e_tp += tp; e_fp += fp; e_fn += fn
            tp, fp, fn = fuzzy_counts(g, p, FUZZY_THRESHOLD, FUZZY_SCORER)
            f_tp += tp; f_fp += fp; f_fn += fn
    _, _, micro_f1 = prf(e_tp, e_fp, e_fn)
    _, _, partial_f1 = prf(f_tp, f_fp, f_fn)
    return {
        "micro_f1": micro_f1,
        "partial_f1": partial_f1,
        "n_samples": len(records),
        "n_parse_fail": n_parse_fail,
    }


# --------------------------------------------------------------------------- #
# Run inventory
# --------------------------------------------------------------------------- #
def model_label(run):
    ma = (run.config.get("cli_configs") or {}).get("model_args") or {}
    suffix = str(ma.get("pretrained") or "?").split("/")[-1]
    return TARGET_MODELS.get(suffix)          # None if not a target model


def strategy_of(run):
    ma = (run.config.get("cli_configs") or {}).get("model_args") or {}
    name = (run.name or "").lower()
    if ma.get("peft") or name.startswith("post"):
        return "lora"
    if "few_shot" in name or "few-shot" in name:
        return "few-shot"
    return "zero-shot"       # includes bare `pre_<ds>` and `pre_zero_shot_`


def _fmt_latex_row(name, values):
    """Format one LaTeX body row: best score \\textbf, second-best \\underline, missing '--'."""
    nums = [v for v in values if v is not None]
    best = max(nums) if nums else None
    second = max([v for v in nums if v < best], default=None) if best is not None else None
    cells = []
    for v in values:
        if v is None:
            cells.append("--")
        else:
            s = f"{v:.3f}"
            if v == best:
                s = r"\textbf{" + s + "}"
            elif second is not None and v == second:
                s = r"\underline{" + s + "}"
            cells.append(s)
    return f"{name} & " + " & ".join(cells) + r" \\"


def write_tables(df, path):
    """Write both a readable wide table and a LaTeX body (per metric) to a text file."""
    lut = {}
    for r in df.itertuples():
        lut[(str(r.dataset), str(r.model), str(r.strategy))] = {
            "micro_f1": r.micro_f1, "partial_f1": r.partial_f1}
    with open(path, "w") as fh:
        for metric, title in [("micro_f1", "MICRO F1"), ("partial_f1", "PARTIAL-MATCH F1")]:
            wide = df.pivot_table(index="dataset", columns=["model", "strategy"],
                                  values=metric, observed=True, aggfunc="first")
            fh.write("=" * 70 + f"\n{title} — readable table\n" + "=" * 70 + "\n")
            fh.write(wide.to_string(float_format=lambda x: f"{x:.3f}") + "\n\n")

            fh.write("=" * 70 + f"\n{title} — LaTeX body rows "
                     "(dataset & Ministral z/f/l & Qwen z/f/l & Gemma z/f/l & GPT z; "
                     "best=\\textbf, 2nd=\\underline)\n" + "=" * 70 + "\n")
            col_accum = {c: [] for c in LATEX_COLS}
            for ds in DATASET_ORDER:
                vals = []
                for c in LATEX_COLS:
                    v = lut.get((ds, c[0], c[1]), {}).get(metric)
                    v = None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)
                    vals.append(v)
                    if v is not None:
                        col_accum[c].append(v)
                fh.write(_fmt_latex_row(ds, vals) + "\n")
            avg = [(sum(col_accum[c]) / len(col_accum[c]) if col_accum[c] else None)
                   for c in LATEX_COLS]
            fh.write(_fmt_latex_row("Average", avg) + "\n\n")
    print(f"tables (readable + LaTeX body) written to: {path}")


def main():
    os.makedirs(CACHE_ROOT, exist_ok=True)
    api = wandb.Api()
    runs = [r for proj in (PROJECT, NEWSPLIT_PROJECT)
            for r in api.runs(f"{ENTITY}/{proj}") if r.state == "finished"]

    # Pass 1: collect ALL candidate runs per (dataset, model, strategy) cell.
    candidates = defaultdict(list)   # cell -> [dict(created, run_id, run_name, artifact), ...]
    for run in runs:
        model = model_label(run)
        if model is None:
            continue
        strat = strategy_of(run)
        for art in run.logged_artifacts():
            if art.type != "samples_by_task":
                continue
            task = art.name.split(":")[0]
            dataset = DATASET_MAP.get(task)
            if dataset is None:
                continue
            candidates[(dataset, model, strat)].append({
                "created": run.created_at,
                "run_id": run.id,
                "run_name": run.name,
                "artifact": art,
            })

    # Pass 2: for each cell, score every candidate and keep the LEAST-DEGENERATE
    # one -- lowest unparseable/empty-output rate, tie-broken by recency. A run
    # can silently degenerate (e.g. the model free-generates an immediate EOS on
    # most inputs -> empty predictions), which teacher-forced eval_loss does not
    # catch; picking most-recent would then surface that broken run over a good
    # earlier one. Ranking by parse-fail rate discards those automatically.
    rows = []
    for cell in sorted(candidates):
        dataset, model, strat = cell
        scored = []
        for info in candidates[cell]:
            art = info["artifact"]
            dl_dir = os.path.join(CACHE_ROOT, info["run_id"], art.name.replace(":", "_"))
            try:
                art.download(root=dl_dir)
                json_files = [f for f in os.listdir(dl_dir) if f.endswith(".json")]
                if not json_files:
                    continue
                records = json.load(open(os.path.join(dl_dir, json_files[0])))
            except Exception as exc:
                print(f"  [skip] {info['run_id']} ({dataset}/{model}/{strat}): {exc}")
                continue
            scores = score_records(records)
            fail_rate = scores["n_parse_fail"] / scores["n_samples"] if scores["n_samples"] else 1.0
            scored.append((fail_rate, info, scores))
        if not scored:
            continue
        # Keep most-recent (the long-standing behaviour), but first drop runs that
        # CATASTROPHICALLY degenerated -- the model free-generates an immediate EOS
        # on most inputs, so a large fraction of predictions are empty/unparseable.
        # Only such collapses (>DEGENERATE_FAIL_RATE) are excluded; ordinary
        # parse-fail noise (a few malformed JSONs) must NOT flip the selection.
        healthy = [t for t in scored if t[0] <= DEGENERATE_FAIL_RATE] or scored
        best = max(healthy, key=lambda t: t[1]["created"])
        _, info, scores = best
        rows.append({
            "dataset": dataset, "model": model, "strategy": strat,
            "micro_f1": round(scores["micro_f1"], 4),
            "partial_f1": round(scores["partial_f1"], 4),
            "n_samples": scores["n_samples"],
            "n_parse_fail": scores["n_parse_fail"],
            "run_id": info["run_id"], "run_name": info["run_name"],
        })
        note = ""
        excluded = [t for t in scored if t[0] > DEGENERATE_FAIL_RATE]
        if excluded:
            note = "  [dropped degenerate: " + ", ".join(f"{t[1]['run_id']}({t[0]:.0%} empty)" for t in excluded) + "]"
        print(f"  [{dataset:>20} | {model:>13} | {strat:<9}] "
              f"micro={scores['micro_f1']:.4f}  partial={scores['partial_f1']:.4f}  "
              f"(n={scores['n_samples']}, parse_fail={scores['n_parse_fail']}, run={info['run_id']}){note}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No matching runs found.")
        return

    # Ordered categoricals for tidy sorting / pivoting.
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df["strategy"] = pd.Categorical(df["strategy"], STRATEGY_ORDER, ordered=True)
    df = df.sort_values(["dataset", "model", "strategy"]).reset_index(drop=True)

    df.to_csv(CSV_OUT, index=False)

    pd.set_option("display.max_rows", None, "display.width", 240,
                  "display.max_columns", None)

    print("\n================= LONG-FORM RESULTS =================")
    print(df[["dataset", "model", "strategy", "micro_f1", "partial_f1",
              "n_samples", "n_parse_fail"]].to_string(index=False))

    for metric in ["micro_f1", "partial_f1"]:
        wide = df.pivot_table(index="dataset", columns=["model", "strategy"],
                              values=metric, observed=True, aggfunc="first")
        print(f"\n================= WIDE TABLE: {metric} =================")
        print(wide.to_string(float_format=lambda x: f"{x:.3f}"))

    # Coverage report: which (dataset, model, strategy) cells are missing.
    have = set((r.dataset, r.model, r.strategy) for r in df.itertuples())
    expected = [(d, m, s) for d in DATASET_ORDER for m in MODEL_ORDER
                for s in STRATEGY_ORDER
                if not (m == "GPT-5.4 mini" and s in ("few-shot", "lora"))]
    missing = [c for c in expected if c not in have]
    print(f"\n================= COVERAGE =================")
    print(f"cells present: {len(have)} / {len(expected)} expected")
    if missing:
        print("MISSING cells:")
        for d, m, s in missing:
            print(f"  - {d} | {m} | {s}")

    write_tables(df, TXT_OUT)

    print(f"\nCSV written to: {CSV_OUT}")
    print(f"Fuzzy config: scorer={FUZZY_SCORER.__name__}, threshold={FUZZY_THRESHOLD}")


if __name__ == "__main__":
    main()
