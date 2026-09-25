import random

import pytest
import torch

from laya.common import DecisionModel
from lavoir.collate import collate_voi, model_inputs
from lavoir.items import make_item
from lavoir.loss import decision_loss, voi_loss
from lavoir.model import LavoirModel, tiny_encoder

from conftest import ENCODER_DIR, BagEncoder, rand_example


def _pair(enc_fn):
    """A DecisionModel and a LavoirModel with the same weights."""
    torch.manual_seed(0)
    voi = LavoirModel(enc_fn()).eval()
    ref = DecisionModel(enc_fn()).eval()
    missing, unexpected = ref.load_state_dict(voi.state_dict(), strict=False)
    assert not missing
    assert set(k.split(".")[0] for k in unexpected) == {"seg_emb", "voi_head", "voi_scale"}
    return ref, voi


def _noslot_batch(tok, n=6, seed=0):
    r = random.Random(seed)
    items = [make_item(tok, rand_example(r, n_slot=0, with_voi=False, idx=i), rng=r) for i in range(n)]
    return collate_voi(items, tok.pad_token_id)


def _compare(ref, voi, b):
    with torch.no_grad():
        l_ref, a_ref = ref(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        out = voi(**{**model_inputs(b, "cpu"), "slot_pos": None})
        out_seg = voi(**model_inputs(b, "cpu"))
    assert torch.equal(l_ref, out["logits"])
    assert torch.equal(a_ref, out["act_logits"])
    assert torch.equal(l_ref, out_seg["logits"])      # with seg_emb = 0, passing seg_ids changes nothing


def test_backward_compat_tiny(tok):
    ref, voi = _pair(lambda: tiny_encoder(len(tok), tok.pad_token_id))
    _compare(ref, voi, _noslot_batch(tok))


@pytest.mark.slow
@pytest.mark.skipif(not ENCODER_DIR, reason="set LAVOIR_ENCODER to a local ModernBERT-large directory")
def test_backward_compat_modernbert_large(tok):
    from transformers import AutoModel
    ref, voi = _pair(lambda: AutoModel.from_pretrained(ENCODER_DIR, attn_implementation="sdpa", dtype=torch.float32))
    _compare(ref, voi, _noslot_batch(tok, n=4))


def _slot_batch(tok, n=8, seed=0, **kw):
    r = random.Random(seed)
    items = [make_item(tok, rand_example(r, idx=i, **kw), rng=r) for i in range(n)]
    return collate_voi(items, tok.pad_token_id)


def test_voi_shapes_and_mask(tok):
    m = LavoirModel(tiny_encoder(len(tok), tok.pad_token_id)).eval()
    b = _slot_batch(tok)
    out = m(**model_inputs(b, "cpu"))
    assert out["voi"].shape == b["slot_mask"].shape
    assert torch.all(out["voi"][~b["slot_mask"]] == 0)


def test_voi_cap_bounded_by_gini(tok):
    """voi_cap: 0 <= VOI * voi_scale <= 1 - sum(p^2); VOI ~ 0 when the decision is certain; masking and the
    gradient separation still hold."""
    torch.manual_seed(0)
    m = LavoirModel(tiny_encoder(len(tok), tok.pad_token_id), voi_cap=True).train()
    m.voi_scale.fill_(0.1)
    b = _slot_batch(tok)
    out = m(**model_inputs(b, "cpu"))
    T = m.temperature[b["qtype"]][:, None]
    gini = 1 - (torch.softmax(out["logits"].detach() / T, -1) ** 2).sum(-1, keepdim=True)
    v = out["voi"] * 0.1
    assert torch.all(v >= 0) and torch.all(v <= gini + 1e-6)
    assert torch.all(out["voi"][~b["slot_mask"]] == 0)
    voi_loss(out["voi"], b["voi_target"], b["voi_mask"], 0.1).backward()
    for n, p in m.scorer.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), n
    m.eval()
    with torch.no_grad():                                   # large logit gaps and a small T: a certain decision
        m.scorer[-1].weight.mul_(1e4)                        # logits ~1e2, well inside the -1e4 mask
        m.temperature.fill_(1e-3)
        sure = m(**model_inputs(b, "cpu"))["voi"]
    assert sure.abs().max() < 1e-3


def test_voi_loss_does_not_touch_decision_logits(tok):
    m = LavoirModel(tiny_encoder(len(tok), tok.pad_token_id)).train()
    b = _slot_batch(tok)
    out = m(**model_inputs(b, "cpu"))
    voi_loss(out["voi"], b["voi_target"], b["voi_mask"], 0.2).backward()
    for n, p in m.scorer.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), n
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters())


def test_permutation_bookkeeping(tok):
    """With a position-free encoder, shuffling the option / slot order must give the same outputs per identity."""
    m = LavoirModel(BagEncoder(len(tok))).eval()
    with torch.no_grad():
        m.seg_emb.weight.normal_()          # also checks that seg_emb lands on the right tokens
    ex = rand_example(random.Random(3), n_opt=5, n_slot=4, idx=0)
    base = make_item(tok, ex)
    outs = []
    for s in range(5):
        it = make_item(tok, ex, rng=random.Random(s))
        b = collate_voi([base, it], tok.pad_token_id)
        with torch.no_grad():
            outs.append(m(**model_inputs(b, "cpu")))
    for o in outs:
        torch.testing.assert_close(o["logits"][0], o["logits"][1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(o["voi"][0], o["voi"][1], atol=1e-4, rtol=1e-4)


def test_decision_loss_runs(tok):
    m = LavoirModel(tiny_encoder(len(tok), tok.pad_token_id)).train()
    b = _slot_batch(tok)
    out = m(**model_inputs(b, "cpu"))
    d = decision_loss(out["logits"], b["target"], b["marker_mask"], b["qtype"], sigma=0.4)
    d["loss"].backward()
    assert torch.isfinite(d["loss"]) and d["kl"] >= -1e-5
