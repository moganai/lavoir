"""On 100 examples the decision KL and the VOI MSE must go to ~0.

Tests the memorization capacity of the whole path (sequence -> model -> loss). The GRPO term is off
(w_rl = 0): with it on, its noisy gradient dominates the shared parameters once the decision has converged
and slows the VOI head down by orders of magnitude.
"""
import random

import torch

from lavoir.collate import collate_voi, model_inputs
from lavoir.items import make_item
from lavoir.loss import decision_loss, voi_loss
from lavoir.model import LavoirModel, tiny_encoder

from conftest import rand_example


def test_overfit_100(tok):
    torch.manual_seed(0)
    r = random.Random(0)
    items = [make_item(tok, rand_example(r, idx=i), rng=r) for i in range(100)]
    b = collate_voi(items, tok.pad_token_id)
    inp = model_inputs(b, "cpu")
    sig_t = float(b["voi_target"][b["voi_mask"]].std())
    m = LavoirModel(tiny_encoder(len(tok), tok.pad_token_id, d=128, layers=2), dropout=0.0)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.0)
    steps = 300
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        out = m(**inp)
        d = decision_loss(out["logits"], b["target"], b["marker_mask"], b["qtype"], sigma=0.1, generator=g, w_rl=0.0)
        lv = voi_loss(out["voi"], b["voi_target"], b["voi_mask"], sig_t)
        opt.zero_grad()
        (d["loss"] + lv).backward()
        opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        out = m(**inp)
        d = decision_loss(out["logits"], b["target"], b["marker_mask"], b["qtype"], sigma=0.1, generator=g)
        lv = voi_loss(out["voi"], b["voi_target"], b["voi_mask"], sig_t)
    assert d["kl"] < 0.05
    assert lv < 0.05
