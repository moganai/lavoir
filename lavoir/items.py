"""JSONL example -> model item.

Example fields: state (str | [{"role", "text"}]), question {"type", "instructions", "criteria"},
target {option: probability}, slots {name: description} (slots already asked are NOT listed),
gold (optional), voi {slot: target} (optional; produced by `lavoir.targets`).
"""
import random
from typing import Dict, List, Optional

from laya.common import QTYPES, render_options

from .sequence import build_sequence_voi, markers_complete


def to_internal(question: Dict) -> Dict:
    t = question["type"]
    crit = question.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    return {"t": t, "ins": question["instructions"], "crit": crit}


def option_keys(q: Dict) -> List[str]:
    if q["t"] == "choice":
        return list(q["crit"].keys())
    if q["t"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(q["crit"]))]


def make_item(tok, ex: Dict, max_len: int = 1024, head_max_len: int = 384, head_max_len_noslot: int = 256,
              rng: Optional[random.Random] = None, state=None, slots: Optional[Dict[str, str]] = None) -> Optional[Dict]:
    """Builds an item; returns None if markers were lost to truncation. `state` / `slots` override the
    example's own (targets.py: state with an appended answer, slot list without the asked slot)."""
    q = to_internal(ex["question"])
    keys = option_keys(q)
    state = ex["state"] if state is None else state
    slots = (ex.get("slots") or {}) if slots is None else slots
    slot_names = list(slots.keys())
    n_opt, n_slot = len(render_options(q)), len(slot_names)
    oorder = list(range(n_opt))
    sorder = list(range(n_slot))
    if rng is not None:
        rng.shuffle(oorder)
        rng.shuffle(sorder)
    seq = build_sequence_voi(tok, state, q, [(s, slots[s]) for s in slot_names], max_len,
                             head_max_len if n_slot else head_max_len_noslot, oorder, sorder,
                             truncate_left=isinstance(state, list))
    if not markers_complete(seq, n_opt, n_slot):
        return None
    it = {"ids": seq["ids"], "option_pos": seq["option_pos"], "slot_pos": seq["slot_pos"],
          "seg_ids": seq["seg_ids"], "qtype": QTYPES[q["t"]], "id": ex.get("id"),
          "workflow": ex.get("workflow"), "option_keys": keys, "slot_names": slot_names}
    if "target" in ex:
        t = [float(ex["target"].get(k, 0.0)) for k in keys]
        s = sum(t)
        t = [v / s for v in t] if s > 0 else [1.0 / len(t)] * len(t)
        it["target"] = t
        it["label"] = max(range(len(t)), key=t.__getitem__)
    if ex.get("gold") is not None:
        it["gold_idx"] = keys.index(str(ex["gold"]))
    if "voi" in ex and n_slot:
        it["voi_target"] = [float(ex["voi"][s]) for s in slot_names]
    return it


def append_turns(state, turns: List[Dict]) -> List[Dict]:
    """Appends turns to a state (a string or a list of turns)."""
    base = [{"role": "user", "text": state}] if isinstance(state, str) else list(state)
    return base + list(turns)
