import warnings

import pytest

from laya.common import build_sequence, render_options
from lavoir.sequence import SEG_OPTION, SEG_SLOT, BudgetWarning, build_sequence_voi, markers_complete

from conftest import rand_text


def _q(n_opt, desc_words=5, r=None):
    crit = {"unit%d" % i: (rand_text(r, desc_words) if r else "desc %d" % i) for i in range(n_opt)}
    return {"t": "choice", "ins": "Which team should handle this request?", "crit": crit}


def _slots(n, r, desc_words=4):
    return [("slot%d" % j, rand_text(r, desc_words)) for j in range(n)]


def test_noslot_identical_to_laya(tok, rng):
    q = _q(4, r=rng)
    state = {"msg": rand_text(rng, 40)}
    order = [2, 0, 3, 1]
    ref_ids, ref_mk = build_sequence(tok, state, q, 1024, 256, order)
    seq = build_sequence_voi(tok, state, q, None, 1024, 256, order)
    assert seq["ids"] == ref_ids
    assert seq["slot_pos"] == []
    # canonical order: option_pos[i] is the marker of option i
    for j, i in enumerate(order):
        assert seq["option_pos"][i] == ref_mk[j]
    for s in ([], {}):
        assert build_sequence_voi(tok, state, q, s, 1024, 256, order)["ids"] == ref_ids


def _check_markers(tok, seq, q, slots):
    ids = seq["ids"]
    assert markers_complete(seq, len(render_options(q)), len(slots))
    assert len(seq["seg_ids"]) == len(ids)
    for p in seq["option_pos"] + seq["slot_pos"]:
        assert ids[p] == tok.mask_token_id
    for p in seq["option_pos"]:
        assert seq["seg_ids"][p] == SEG_OPTION
    for p in seq["slot_pos"]:
        assert seq["seg_ids"][p] == SEG_SLOT
    # canonical mapping: the text after each marker starts with the right option / slot name
    opts = render_options(q)
    for i, p in enumerate(seq["option_pos"]):
        nxt = tok.decode(ids[p + 1: p + 4]).strip()
        assert opts[i].startswith(nxt[:4]) or nxt.startswith(opts[i][:4]), (i, nxt, opts[i])
    for j, p in enumerate(seq["slot_pos"]):
        nxt = tok.decode(ids[p + 1: p + 5]).strip()
        assert nxt.startswith(slots[j][0][:4]), (j, nxt, slots[j][0])


CASES = {
    "normal": dict(n_opt=4, n_slot=4, state_words=40, slot_desc=4, head=384),
    "long_state": dict(n_opt=4, n_slot=4, state_words=3000, slot_desc=4, head=384),
    "50_options": dict(n_opt=50, n_slot=4, state_words=40, slot_desc=4, head=384),
    "8_slots": dict(n_opt=6, n_slot=8, state_words=200, slot_desc=6, head=384),
    "long_slot_descriptions": dict(n_opt=4, n_slot=5, state_words=40, slot_desc=200, head=384),
    "everything_long": dict(n_opt=50, n_slot=8, state_words=3000, slot_desc=200, head=384),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("shuffle", [False, True])
def test_markers_point_to_mask(tok, rng, name, shuffle):
    c = CASES[name]
    q = _q(c["n_opt"], r=rng)
    slots = _slots(c["n_slot"], rng, c["slot_desc"])
    state = [{"role": "user", "text": rand_text(rng, c["state_words"])}]
    oo = list(range(c["n_opt"]))
    so = list(range(c["n_slot"]))
    if shuffle:
        rng.shuffle(oo)
        rng.shuffle(so)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BudgetWarning)
        seq = build_sequence_voi(tok, state, q, slots, 1024, c["head"], oo, so, truncate_left=True)
    assert len(seq["ids"]) <= 1024
    _check_markers(tok, seq, q, slots)


def test_slot_token_limits(tok, rng):
    q = _q(3, r=rng)
    slots = _slots(3, rng, desc_words=200)
    seq = build_sequence_voi(tok, "x", q, slots, 1024, 384)
    sp = seq["slot_pos"] + [seq["ids"].index(tok.sep_token_id, seq["slot_pos"][-1])]
    for a, b in zip(sp[:-1], sp[1:]):
        assert b - a <= 24          # cap, marker included


def test_budget_warning(tok, rng):
    q = _q(50, desc_words=30, r=rng)
    slots = _slots(8, rng, 30)
    with pytest.warns(BudgetWarning):
        seq = build_sequence_voi(tok, "hello", q, slots, 1024, 128)
    _check_markers(tok, seq, q, slots)       # a warning, but no lost marker


def test_no_warning_when_fits(tok, rng):
    q = _q(5, r=rng)
    with warnings.catch_warnings():
        warnings.simplefilter("error", BudgetWarning)
        build_sequence_voi(tok, "hello", q, _slots(6, rng), 1024, 384)


def test_truncate_left_keeps_last_turn(tok, rng):
    q = _q(3, r=rng)
    state = [{"role": "user", "text": rand_text(rng, 3000)}, {"role": "user", "text": "LASTTURN marker"}]
    seq = build_sequence_voi(tok, state, q, _slots(3, rng), 512, 384, truncate_left=True)
    assert "LASTTURN" in tok.decode(seq["ids"])


def test_marker_loss_detected(tok, rng):
    q = _q(50, desc_words=30, r=rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BudgetWarning)
        seq = build_sequence_voi(tok, "x", q, _slots(8, rng, 30), 128, 1000)
    assert not markers_complete(seq, 50, 8)


def test_bad_order_rejected(tok, rng):
    with pytest.raises(ValueError):
        build_sequence_voi(tok, "x", _q(3, r=rng), _slots(2, rng), 1024, 384, [0, 0, 1])
