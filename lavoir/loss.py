"""Losses.

L = L_decision + lambda * MSE(voi[voi_mask], target / sigma_target)

L_decision is the same as in the Laya training notebook (`train_ddp.py`): noisy-logit GRPO (reward =
`proper_reward`, group-mean baseline, zero-mean noise) plus full-weight soft cross-entropy. The GRPO term is
weighted by `w_rl`; the released model was trained with w_rl = 0 (see the README for why).
"""
from typing import Dict, Optional

import torch

from laya.common import proper_reward


def decision_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, qtype: torch.Tensor,
                  sigma: float, group_size: int = 4, w_ce: float = 1.0, w_rl: float = 1.0,
                  generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, device=logits.device, generator=generator) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    logq = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
    loss_ce = -(target * logq).sum(-1).mean()
    # KL(target || p): with soft targets the reachable minimum of the CE is the entropy, so KL is logged
    ent = -(target * torch.log(target.clamp_min(1e-12))).sum(-1).mean()
    return {"loss": w_rl * loss_rl + w_ce * loss_ce, "rl": loss_rl, "ce": loss_ce, "kl": loss_ce - ent,
            "reward": r.mean()}


def voi_loss(voi: torch.Tensor, voi_target: torch.Tensor, voi_mask: torch.Tensor, sigma_t: float) -> torch.Tensor:
    """MSE against the target normalized by sigma_target. Returns 0 (attached to the graph) for an empty mask."""
    if not voi_mask.any():
        return voi.sum() * 0.0
    d = voi[voi_mask] - voi_target[voi_mask] / sigma_t
    return (d ** 2).mean()
