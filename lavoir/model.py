"""LavoirModel: Laya's DecisionModel + a segment embedding + a per-slot VOI head."""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from laya.common import DecisionModel, _apply_rope_config


def decision_feats(logits: torch.Tensor, marker_mask: torch.Tensor) -> torch.Tensor:
    """[top1, top1 - top2, normalized entropy, k/255]: the same features Laya's act_head uses."""
    p = torch.softmax(logits, -1)
    k = marker_mask.sum(-1).clamp(min=2).float()
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
    if p.size(-1) >= 2:
        top2 = p.topk(2, -1).values
    else:
        top1 = p.topk(1, -1).values
        top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
    return torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)


class LavoirModel(DecisionModel):
    """DecisionModel + seg_emb (zero-initialized) + voi_head.

    seg_emb is added to the encoder OUTPUT, where type_emb is added. Because it starts at zero, a slot-free
    input gives exactly the logits of a DecisionModel with the same weights.
    VOI head input: the hidden state at the slot marker + detached decision features (softmax(logits / T)),
    so the VOI loss sends no gradient into the decision logits.

    voi_cap=True: VOI = Gini(p) * sigmoid(h), in units of p(gold) (the output is still divided by voi_scale).
    For a calibrated model the expected gain in p(gold) is bounded by 1 - sum(p^2), so a confident model
    (p close to one-hot) structurally predicts VOI close to 0. Without the cap the head can predict large VOI
    for confident decisions on out-of-distribution inputs, which leads to unnecessary questions.
    """

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1,
                 voi_hidden: int = 256, voi_cap: bool = False):
        super().__init__(encoder, head_layers, n_act, dropout)
        self.voi_cap = voi_cap
        d = encoder.config.hidden_size
        self.seg_emb = nn.Embedding(3, d)
        nn.init.zeros_(self.seg_emb.weight)
        self.voi_head = nn.Sequential(nn.LayerNorm(d + 4), nn.Linear(d + 4, voi_hidden), nn.GELU(),
                                      nn.Linear(voi_hidden, 1))
        # VOI target scale: the head predicts sigma_target-normalized values; voi_scale converts to p(gold) units.
        self.register_buffer("voi_scale", torch.ones(()))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
                seg_ids: Optional[torch.Tensor] = None, slot_pos: Optional[torch.Tensor] = None,
                slot_mask: Optional[torch.Tensor] = None, detach_encoder: bool = False) -> Dict[str, torch.Tensor]:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if seg_ids is not None:
            h = h + self.seg_emb(seg_ids)
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = self.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        feats_raw = decision_feats(logits.detach(), marker_mask)
        act_logits = self.act_head(torch.cat([h[:, 0].float(), feats_raw], -1))
        out = {"logits": logits, "act_logits": act_logits, "voi": None}

        if slot_pos is not None and slot_pos.size(1) > 0:
            T = self.temperature[qtype].clamp_min(1e-3)[:, None]
            zT = logits.detach() / T
            feats = decision_feats(zT, marker_mask)                                 # [B, 4]
            sidx = slot_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            hs = torch.gather(h, 1, sidx).float()                                   # [B, S, d]
            f = feats[:, None, :].expand(-1, hs.size(1), -1)
            voi = self.voi_head(torch.cat([hs, f], -1)).squeeze(-1).float()        # [B, S]
            if self.voi_cap:
                gini = 1.0 - (torch.softmax(zT, -1) ** 2).sum(-1, keepdim=True)     # [B, 1]
                voi = gini * torch.sigmoid(voi) / self.voi_scale
            out["voi"] = voi.masked_fill(~slot_mask, 0.0) if slot_mask is not None else voi
        return out


def n_act_of(cfg: Dict) -> int:
    return int(cfg.get("n_act", len(cfg.get("act_costs", {})) + 1))


def build_voi_model(cfg: Dict, encoder_dir: Optional[str] = None, pretrained: bool = True) -> LavoirModel:
    """VOI counterpart of laya.common.build_model. pretrained=True loads encoder weights; the heads are random."""
    from transformers import AutoConfig, AutoModel

    src = encoder_dir or cfg["encoder"]
    if pretrained:
        enc = AutoModel.from_pretrained(src, attn_implementation="sdpa")
    else:
        ecfg = AutoConfig.from_pretrained(src)
        _apply_rope_config(ecfg)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    return LavoirModel(enc, cfg.get("head_layers", 2), n_act_of(cfg), voi_cap=cfg.get("voi_cap", False))


def tiny_encoder(vocab: int, pad_id: int, d: int = 64, layers: int = 2, seed: int = 0):
    """A small random ModernBERT for smoke tests (`--tiny`) and unit tests."""
    from transformers import ModernBertConfig, ModernBertModel
    torch.manual_seed(seed)
    c = ModernBertConfig(vocab_size=vocab, hidden_size=d, intermediate_size=2 * d, num_hidden_layers=layers,
                         num_attention_heads=max(1, d // 32), pad_token_id=pad_id, max_position_embeddings=2048)
    return ModernBertModel._from_config(c, attn_implementation="sdpa")
