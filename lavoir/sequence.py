"""Input sequence: the Laya format plus a block of candidate questions ("slots").

    [CLS] <type> question: <instruction> [SEP]
    [MASK] <option1>: <description> [MASK] <option2>: ... [SEP]
    missing information: [MASK] <slot1>: <description> [MASK] <slot2>: ... [SEP]
    <state> [SEP]

Without slots, `laya.common.build_sequence` is called unchanged, so a VOI model reads slot-free
inputs exactly like Laya does.

The returned `option_pos` / `slot_pos` are in CANONICAL order: `option_pos[i]` is the marker of
`render_options(q)[i]`, `slot_pos[j]` is the marker of the j-th entry of `slots`. Shuffling
(`option_order` / `slot_order`) only changes where the blocks sit in the token sequence; targets never
need to be reordered.
"""
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

from laya.common import build_sequence, render_options, serialize_state

SLOT_HEADER = "missing information:"
OPT_CAP = 48          # Laya: an option text is at most 48 tokens (+ marker)
OPT_MIN = 4           # per-option floor (marker included)
SLOT_CAP = 24         # per-slot cap (marker included)
SLOT_MIN = 8          # per-slot floor (marker included)
INS_MIN = 8           # floor for the instruction
INS_RESERVE = 16      # Laya: room kept for the instruction while options are compressed

SEG_OTHER, SEG_OPTION, SEG_SLOT = 0, 1, 2


class BudgetWarning(UserWarning):
    """The option and slot floors do not fit in head_max_len; the head section exceeds its budget."""


def _invert(order: Sequence[int], pos_in_order: List[int]) -> List[int]:
    out = [0] * len(order)
    for j, i in enumerate(order):
        out[i] = pos_in_order[j]
    return out


def _check_order(order, n, name):
    if sorted(order) != list(range(n)):
        raise ValueError("%s must be a permutation: %r (n=%d)" % (name, order, n))


def build_sequence_voi(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    slots: Optional[Sequence[Tuple[str, str]]] = None,
    max_len: int = 1024,
    head_max_len: int = 384,
    option_order: Optional[List[int]] = None,
    slot_order: Optional[List[int]] = None,
    truncate_left: bool = False,
) -> Dict[str, List[int]]:
    """Builds the token sequence.

    q: Laya internal format {"t", "ins", "crit"[, "labels"]}.
    slots: [(name, description), ...] in canonical order; empty/None gives the plain Laya format.
    Returns {"ids", "option_pos", "slot_pos", "seg_ids"}. Markers lost to truncation are dropped from the
    lists; callers must check the lengths (`markers_complete`).
    """
    mask_tok = tok.mask_token
    n_opt = len(render_options(q))
    order = list(option_order) if option_order is not None else list(range(n_opt))
    _check_order(order, n_opt, "option_order")

    if not slots:
        ids, markers = build_sequence(tok, state, q, max_len, head_max_len, order, truncate_left)
        option_pos = _invert(order, markers) if len(markers) == n_opt else markers
        seg = [SEG_OTHER] * len(ids)
        if markers:
            # markers increase in token order; the option block runs up to the first [SEP] after the last marker
            try:
                end = ids.index(tok.sep_token_id, markers[-1])
            except ValueError:          # max_len cut the block in the middle
                end = len(ids)
            for p in range(markers[0], end):
                seg[p] = SEG_OPTION
        return {"ids": ids, "option_pos": option_pos, "slot_pos": [], "seg_ids": seg}

    n_slot = len(slots)
    sorder = list(slot_order) if slot_order is not None else list(range(n_slot))
    _check_order(sorder, n_slot, "slot_order")

    opts = render_options(q)
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = [[tok.mask_token_id]
               + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:OPT_CAP]
               for i in order]
    slot_ids = []
    for j in sorder:
        name, desc = slots[j]
        txt = ("%s: %s" % (name, desc) if desc else str(name)).replace(mask_tok, " ")
        slot_ids.append([tok.mask_token_id]
                        + tok(" " + txt, add_special_tokens=False)["input_ids"][: SLOT_CAP - 1])
    hdr_ids = tok(SLOT_HEADER, add_special_tokens=False)["input_ids"]

    # Budget: head_max_len = instruction + options + slot header + slots ([CLS]/[SEP] excluded, as in Laya).
    avail = head_max_len - len(hdr_ids)
    slot_floor = sum(min(len(s), SLOT_MIN) for s in slot_ids)
    # 1) Options first: shortened down to OPT_MIN per option if needed, without touching the slot floor.
    if sum(len(o) for o in opt_ids) + slot_floor + INS_RESERVE > avail:
        per = max(OPT_MIN, (avail - INS_RESERVE - slot_floor) // max(1, n_opt))
        opt_ids = [o[:per] for o in opt_ids]
    # 2) Then slots: shortened down to SLOT_MIN per slot if needed.
    rem = avail - sum(len(o) for o in opt_ids) - INS_RESERVE
    if sum(len(s) for s in slot_ids) > rem:
        per_s = max(SLOT_MIN, rem // n_slot)
        slot_ids = [s[:per_s] for s in slot_ids]
    # 3) The instruction takes what is left (at least INS_MIN).
    ins_room = avail - sum(len(o) for o in opt_ids) - sum(len(s) for s in slot_ids)
    head_ids = head_ids[: max(INS_MIN, ins_room)]
    used = len(head_ids) + len(hdr_ids) + sum(len(o) for o in opt_ids) + sum(len(s) for s in slot_ids)
    if used > head_max_len:
        warnings.warn(
            "head budget exceeded: %d > head_max_len=%d (%d options, %d slots); the floors do not fit, "
            "increase head_max_len" % (used, head_max_len, n_opt, n_slot), BudgetWarning, stacklevel=2)

    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    seg = [SEG_OTHER] * len(ids)
    omark = []
    for o in opt_ids:
        omark.append(len(ids))
        ids.extend(o)
        seg.extend([SEG_OPTION] * len(o))
    ids.append(tok.sep_token_id)
    seg.append(SEG_OTHER)
    ids.extend(hdr_ids)
    seg.extend([SEG_SLOT] * len(hdr_ids))
    smark = []
    for s in slot_ids:
        smark.append(len(ids))
        ids.extend(s)
        seg.extend([SEG_SLOT] * len(s))
    ids.append(tok.sep_token_id)
    seg.append(SEG_OTHER)

    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[max(0, len(st) - room):] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    seg = seg + [SEG_OTHER] * (len(st) + 1)
    ids, seg = ids[:max_len], seg[:max_len]

    option_pos = _invert(order, omark) if all(m < max_len for m in omark) else [m for m in omark if m < max_len]
    slot_pos = _invert(sorder, smark) if all(m < max_len for m in smark) else [m for m in smark if m < max_len]
    return {"ids": ids, "option_pos": option_pos, "slot_pos": slot_pos, "seg_ids": seg}


def markers_complete(seq: Dict[str, List[int]], n_opt: int, n_slot: int) -> bool:
    return len(seq["option_pos"]) == n_opt and len(seq["slot_pos"]) == n_slot
