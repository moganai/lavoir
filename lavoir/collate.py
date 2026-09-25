"""Batching: Laya's collate_items plus the slot fields.

Item fields:
  ids, option_pos, qtype            required
  seg_ids                           all zeros if missing
  slot_pos                          no slots if missing
  target      [n_opt]               decision target (probabilities over options)
  voi_target  [n_slot]              VOI target (in units of p(gold), NOT normalized)
  label                             argmax of the target (-1 if missing)
"""
from typing import Dict, List

import torch

META_SKIP = ("ids", "option_pos", "slot_pos", "seg_ids", "target", "voi_target")


def collate_voi(items: List[Dict], pad_id: int) -> Dict:
    if not items:
        raise ValueError("collate_voi: empty batch")
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["option_pos"]) for it in items)
    smax = max(len(it.get("slot_pos", [])) for it in items)

    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    seg = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    spos = torch.zeros((n, smax), dtype=torch.long)
    smask = torch.zeros((n, smax), dtype=torch.bool)
    has_t = any("target" in it for it in items)
    has_v = any("voi_target" in it for it in items)
    target = torch.zeros((n, kmax)) if has_t else None
    voi_t = torch.zeros((n, smax)) if has_v else None

    for i, it in enumerate(items):
        m = len(it["ids"])
        ids[i, :m] = torch.tensor(it["ids"])
        att[i, :m] = 1
        if "seg_ids" in it:
            if len(it["seg_ids"]) != m:
                raise ValueError("item %d: seg_ids length %d != ids %d" % (i, len(it["seg_ids"]), m))
            seg[i, :m] = torch.tensor(it["seg_ids"])
        k = len(it["option_pos"])
        mpos[i, :k] = torch.tensor(it["option_pos"])
        mmask[i, :k] = True
        s = len(it.get("slot_pos", []))
        if s:
            spos[i, :s] = torch.tensor(it["slot_pos"])
            smask[i, :s] = True
        if has_t and "target" in it:
            if len(it["target"]) != k:
                raise ValueError("item %d: %d targets for %d option markers" % (i, len(it["target"]), k))
            target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
        if has_v and "voi_target" in it:
            if len(it["voi_target"]) != s:
                raise ValueError("item %d: %d VOI targets for %d slot markers" % (i, len(it["voi_target"]), s))
            voi_t[i, :s] = torch.tensor(it["voi_target"], dtype=torch.float32)

    res = {
        "input_ids": ids, "attention_mask": att, "seg_ids": seg,
        "marker_pos": mpos, "marker_mask": mmask,
        "slot_pos": spos, "slot_mask": smask,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it.get("label", -1) for it in items]),
        "meta": [{k: v for k, v in it.items() if k not in META_SKIP} for it in items],
    }
    if target is not None:
        res["target"] = target
    if voi_t is not None:
        res["voi_target"] = voi_t
        # slots of items without a VOI target stay out of the loss
        res["voi_mask"] = smask & torch.tensor([("voi_target" in it) for it in items])[:, None]
    return res


MODEL_KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "seg_ids", "slot_pos", "slot_mask")


def model_inputs(batch: Dict, device) -> Dict[str, torch.Tensor]:
    return {k: batch[k].to(device) for k in MODEL_KEYS}
