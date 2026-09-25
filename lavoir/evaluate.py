"""Evaluation on held-out JSONL data.

    python -m lavoir.evaluate --checkpoint runs/joint_v2 --data data/test.jsonl --split test --out runs/eval

Decision: accuracy against the target's argmax (and against `gold` when present), expected calibration error,
KL(target || p); overall and per source (`workflow` field).
VOI (examples with `probes` and `gold`): the realized gain of each slot, p(gold | state + question + sampled
answer) - p(gold | state), is computed with the same model and compared with the predicted VOI:
Spearman correlation, mean absolute error, and how often the slot with the highest predicted VOI is also the one
with the highest realized gain. A single sampled answer per slot makes the realized gain noisy, so these numbers
are lower bounds on how well VOI tracks the expected gain.
Raw per-example values go to `decisions.jsonl` and `voi.jsonl`, the summary to `summary.json`.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from laya.common import ece_score

from .api import Lavoir
from .targets import compute_voi_targets
from .train import load_examples


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def decision_rows(model: Lavoir, exs, calibration_source=None):
    preds = model.predict_batch([(e["state"], e["question"], e.get("slots")) for e in exs], calibration_source)
    rows = []
    for e, pr in zip(exs, preds):
        keys = list(pr.probabilities)
        p = np.array([pr.probabilities[k] for k in keys])
        t = np.array([float(e["target"].get(k, 0.0)) for k in keys])
        t = t / t.sum() if t.sum() > 0 else np.full(len(t), 1 / len(t))
        row = {"id": e["id"], "source": e.get("workflow"), "pred": keys[int(p.argmax())], "confidence": float(p.max()),
               "correct": float(p.argmax() == t.argmax()),
               "kl": float((t * (np.log(np.clip(t, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1)))).sum()),
               "p": pr.probabilities, "voi": pr.voi}
        if e.get("gold") is not None:
            row["gold"] = e["gold"]
            row["correct_gold"] = float(row["pred"] == str(e["gold"]))
        rows.append(row)
    return rows


def summarize(rows):
    def one(rs):
        a = np.array([r["correct"] for r in rs])
        c = np.array([r["confidence"] for r in rs])
        out = {"n": len(rs), "acc": round(float(a.mean()), 4), "ece": round(ece_score(c, a), 4),
               "kl": round(float(np.mean([r["kl"] for r in rs])), 4)}
        g = [r for r in rs if "correct_gold" in r]
        if g:
            ag = np.array([r["correct_gold"] for r in g])
            out.update(acc_gold=round(float(ag.mean()), 4),
                       ece_gold=round(ece_score(np.array([r["confidence"] for r in g]), ag), 4))
        return out
    by = defaultdict(list)
    for r in rows:
        by[str(r["source"])].append(r)
    return {"_all": one(rows), **{k: one(v) for k, v in sorted(by.items())}}


def voi_quality(model: Lavoir, exs, pred_rows, batch_size=64):
    ex_voi = [e for e in exs if e.get("probes") and e.get("gold") is not None and e.get("slots")]
    if not ex_voi:
        return {}, []
    amp = torch.bfloat16 if model.device.type == "cuda" else None
    realized = compute_voi_targets(model.model, model.tokenizer, ex_voi, model.device, model.max_len,
                                   model.head_max_len, batch_size, amp)
    pred_by_id = {r["id"]: r["voi"] for r in pred_rows}
    xs, ys, top1, raw = [], [], [], []
    for r in realized:
        pv = pred_by_id.get(r["id"]) or {}
        slots = [s for s in r["voi"] if s in pv]
        for s in slots:
            xs.append(pv[s])
            ys.append(r["voi"][s])
        if slots and max(abs(r["voi"][s]) for s in slots) > 1e-6:
            top1.append(float(max(slots, key=pv.get) == max(slots, key=r["voi"].get)))
        raw.append({"id": r["id"], "predicted": pv, "realized": r["voi"], "p0_gold": r["p0_gold"]})
    summ = {"n_slots": len(xs), "spearman": round(spearman(xs, ys), 4),
            "mae": round(float(np.mean(np.abs(np.array(xs) - np.array(ys)))), 5) if xs else None,
            "top1_agreement": round(float(np.mean(top1)), 4) if top1 else None}
    return summ, raw


def main():
    ap = argparse.ArgumentParser(description="Evaluate a Lavoir checkpoint on JSONL data.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", help="keep examples whose \"split\" field equals this")
    ap.add_argument("--calibration_source", default=None, help="voi, general or a training source name")
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = Lavoir.from_pretrained(a.checkpoint, batch_size=a.batch_size)
    exs = load_examples(a.data, a.split)
    if not exs:
        raise SystemExit("no examples with split=%r in %s" % (a.split, a.data))
    rows = decision_rows(model, exs, a.calibration_source)
    summary = {"checkpoint": a.checkpoint, "data": a.data, "decision": summarize(rows)}
    vq, vraw = voi_quality(model, exs, rows, a.batch_size)
    if vq:
        summary["voi"] = vq
    with open(os.path.join(a.out, "decisions.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "voi.jsonl"), "w") as f:
        for r in vraw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({"decision": summary["decision"]["_all"], "voi": summary.get("voi")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
