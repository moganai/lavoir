"""VOI targets from a model snapshot.

    p0 = softmax(logits(state) / T)
    for each slot k not asked yet:
        pk = softmax(logits(state + question_k + answer_k) / T)
        target[k] = pk[gold] - p0[gold]          # no clipping

Every example needs `gold` and `probes: {slot: {"q": question, "a": answer}}`, one sampled answer per slot.
In the sequence for pk the slot list is the remaining slots minus k (what the model would see at that point).
The state is always serialized as a list of turns so that p0 and pk use the same format.
Raw values (p0, p(gold) before and after) are written to the output.

    python -m lavoir.targets --checkpoint runs/decision --data data/train.jsonl --out runs/voi_targets.jsonl
"""
import argparse
import json
from typing import Dict, Iterable, List

import torch

from .collate import collate_voi, model_inputs
from .items import append_turns, make_item


@torch.no_grad()
def _probs(model, items: List[Dict], pad_id: int, device, batch_size: int, amp_dtype=None) -> List[List[float]]:
    out = []
    for b in range(0, len(items), batch_size):
        chunk = items[b:b + batch_size]
        batch = collate_voi(chunk, pad_id)
        inp = model_inputs(batch, device)
        inp["slot_pos"] = None          # the VOI head is not needed here
        if amp_dtype is not None:
            with torch.autocast(device.type, dtype=amp_dtype):
                logits = model(**inp)["logits"]
        else:
            logits = model(**inp)["logits"]
        T = model.temperature[inp["qtype"]].clamp_min(1e-3)[:, None]
        p = torch.softmax(logits.float() / T, -1).cpu()
        for i, it in enumerate(chunk):
            out.append(p[i, : len(it["option_pos"])].tolist())
    return out


def compute_voi_targets(model, tok, examples: Iterable[Dict], device, max_len=1024, head_max_len=384,
                        batch_size=32, amp_dtype=None) -> List[Dict]:
    model.eval()
    jobs, meta = [], []        # (example index, slot or None)
    exs = list(examples)
    for n, ex in enumerate(exs):
        if ex.get("gold") is None:
            raise ValueError("example %r has no gold label; its VOI target cannot be computed" % ex.get("id"))
        slots = dict(ex.get("slots") or {})
        base = append_turns(ex["state"], [])
        it0 = make_item(tok, ex, max_len, head_max_len, state=base, slots=slots)
        if it0 is None:
            continue
        jobs.append(it0)
        meta.append((n, None))
        for k in slots:
            if k not in ex.get("probes", {}):
                raise ValueError("example %r: no probe (sampled answer) for slot %r" % (ex.get("id"), k))
            pr = ex["probes"][k]
            st = append_turns(base, [{"role": "system", "text": pr["q"]}, {"role": "user", "text": pr["a"]}])
            rest = {s: d for s, d in slots.items() if s != k}
            itk = make_item(tok, ex, max_len, head_max_len, state=st, slots=rest)
            if itk is None:
                raise ValueError("example %r slot %r: markers lost to truncation" % (ex.get("id"), k))
            jobs.append(itk)
            meta.append((n, k))

    probs = _probs(model, jobs, tok.pad_token_id, device, batch_size, amp_dtype)
    res: Dict[int, Dict] = {}
    for (n, k), it, p in zip(meta, jobs, probs):
        g = it["gold_idx"]
        if k is None:
            res[n] = {"id": exs[n].get("id"), "p0": p, "p0_gold": p[g], "voi": {}, "pk_gold": {}}
        else:
            res[n]["pk_gold"][k] = p[g]
    for r in res.values():
        r["voi"] = {k: v - r["p0_gold"] for k, v in r["pk_gold"].items()}
    return [res[n] for n in sorted(res)]


def main():
    from .checkpoint import load_checkpoint
    from .train import load_examples

    ap = argparse.ArgumentParser(description="Compute VOI training targets with a model snapshot.")
    ap.add_argument("--checkpoint", required=True, help="checkpoint directory or Hugging Face Hub repo id")
    ap.add_argument("--data", nargs="+", required=True, help="JSONL files (glob patterns allowed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0, help="for parallel runs: this process takes examples shard::num_shards")
    ap.add_argument("--num_shards", type=int, default=1)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_checkpoint(a.checkpoint, device)
    exs = [e for e in load_examples(a.data, a.split) if e.get("probes")][a.shard::a.num_shards]
    res = compute_voi_targets(model, tok, exs, device, cfg.get("max_len", 1024), cfg.get("head_max_len", 384),
                              a.batch_size, torch.bfloat16 if device.type == "cuda" else None)
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vals = [v for r in res for v in r["voi"].values()]
    print(json.dumps({"examples": len(res), "slots": len(vals), "mean_voi": sum(vals) / max(1, len(vals)),
                      "share_above_0.05": sum(v > 0.05 for v in vals) / max(1, len(vals)),
                      "temperature": model.temperature.tolist()}))


if __name__ == "__main__":
    main()
