"""Training and fine-tuning.

Phases
  decision     decision head (+ encoder) on single-turn data; the VOI head is off.
               From scratch: --encoder answerdotai/ModernBERT-large (a frozen-encoder head warm-up comes first).
               Fine-tuning: --init <checkpoint>.
  joint        decision loss + lambda * VOI loss, low encoder LR. Needs --init and --voi_targets
               (from `python -m lavoir.targets`).
  temperature  no training: reload --init and refit the calibration temperatures on the held-out slice.

    python -m lavoir.train --phase decision --encoder answerdotai/ModernBERT-large \
        --data data/*.jsonl --out runs/decision
    python -m lavoir.targets --checkpoint runs/decision --data data/*.jsonl --out runs/voi_targets_v1.jsonl
    python -m lavoir.train --phase joint --init runs/decision --voi_targets runs/voi_targets_v1.jsonl \
        --data data/*.jsonl --out runs/joint_v1 --epochs 2

Multi-GPU: launch with torchrun; --batch_size is PER GPU (global = batch_size x world size).

Loss: lavoir.loss.decision_loss (noisy-logit GRPO + soft CE; sigma goes 0.4 -> 0.1 over training; the GRPO
term is weighted by --w_rl, default 0) [+ lambda * lavoir.loss.voi_loss]. bf16 autocast on CUDA.
Calibration: a held-out slice (--calib_frac, never trained on) is used for per-epoch metrics and to fit one
temperature per question type, separately for VOI workflows (examples with `probes`), for general data and
for every training source (`workflow` field).
"""
import argparse
import glob
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from laya.common import QTYPE_NAMES, ece_score

from .checkpoint import load_checkpoint, load_tokenizer, save_checkpoint
from .collate import collate_voi, model_inputs
from .items import make_item
from .loss import decision_loss, voi_loss
from .model import LavoirModel, build_voi_model, tiny_encoder


# ------------------------------------------------------------------ data
def load_examples(paths, split="train"):
    """Reads JSONL files (glob patterns allowed). Lines without `question` and `target` are skipped;
    `split` filters on the example's "split" field (default "train"). Missing ids are filled in."""
    exs = []
    for p in paths:
        files = sorted(glob.glob(p))
        if not files:
            raise FileNotFoundError("no file matches %r" % p)
        for f in files:
            with open(f) as fh:
                for n, line in enumerate(fh):
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    if "question" not in e or "target" not in e:
                        continue
                    if e.get("split", "train") == split:
                        e.setdefault("id", "%s:%d" % (os.path.basename(f), n))
                        exs.append(e)
    return exs


class ExDataset(Dataset):
    """Option and slot order are reshuffled on every access."""

    def __init__(self, exs, tok, max_len, head_max_len, voi=None, seed=0, shuffle=True):
        self.exs, self.tok, self.max_len, self.hml = exs, tok, max_len, head_max_len
        self.voi, self.seed, self.shuffle, self.epoch = voi or {}, seed, shuffle, 0

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, i):
        e = self.exs[i]
        if e["id"] in self.voi:
            e = {**e, "voi": self.voi[e["id"]]}
        rng = random.Random(hash((self.seed, self.epoch, i))) if self.shuffle else None
        it = make_item(self.tok, e, self.max_len, self.hml, rng=rng)
        if it is not None:
            it["src"] = e.get("workflow")
            it["gold_key"] = e.get("gold")
        return it


def est_len(e):
    q = e["question"]
    return len(json.dumps(e["state"], ensure_ascii=False)) + len(json.dumps(q.get("criteria"), ensure_ascii=False)) \
        + 40 * len(e.get("slots") or {})


def length_batches(exs, bs, seed, mega=50):
    """Shuffle -> sort by length inside mega-groups -> batches -> shuffle the batch order."""
    r = random.Random(seed)
    idx = list(range(len(exs)))
    r.shuffle(idx)
    out = []
    for k in range(0, len(idx), bs * mega):
        chunk = sorted(idx[k:k + bs * mega], key=lambda i: est_len(exs[i]))
        out += [chunk[j:j + bs] for j in range(0, len(chunk), bs)]
    r.shuffle(out)
    return out


def make_collate(pad_id):
    def f(items):
        items = [it for it in items if it is not None]
        return collate_voi(items, pad_id) if items else None
    return f


# ------------------------------------------------------------------ temperature and metrics
def fit_one_temp(sel):
    """One temperature minimizing soft CE (LBFGS), as in the Laya notebook. Fewer than 10 rows -> 1.0."""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    rows = []
    for b in loader:
        if b is None:
            continue
        inp = model_inputs(b, device)
        if not b["slot_mask"].any().item():
            inp["slot_pos"] = None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(**inp)
        lg = out["logits"].float().cpu()
        vo = out["voi"].float().cpu() if out["voi"] is not None else None
        for i, m in enumerate(b["meta"]):
            k = int(b["marker_mask"][i].sum())
            r = {"id": m["id"], "src": m.get("src"), "qtype": int(b["qtype"][i]), "logits": lg[i, :k].tolist(),
                 "target": b["target"][i, :k].tolist(), "option_keys": m["option_keys"]}
            if vo is not None and b["slot_mask"][i].any():
                s = int(b["slot_mask"][i].sum())
                r["voi"] = dict(zip(m["slot_names"], vo[i, :s].tolist()))
                if "voi_target" in b:
                    r["voi_target"] = dict(zip(m["slot_names"], b["voi_target"][i, :s].tolist()))
            rows.append(r)
    return rows


def metrics(rows, temps):
    by = defaultdict(list)
    for r in rows:
        z = np.array(r["logits"]) / temps[r["qtype"]]
        p = np.exp(z - z.max())
        p /= p.sum()
        t = np.array(r["target"])
        kl = float((t * (np.log(np.clip(t, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1)))).sum())
        rec = (float(p.argmax() == t.argmax()), float(p.max()), kl)
        by[r["src"]].append(rec)
        by["_all"].append(rec)
    out = {}
    for s, v in by.items():
        a = np.array(v)
        out[str(s)] = {"n": len(v), "acc": round(float(a[:, 0].mean()), 4), "ece": round(ece_score(a[:, 1], a[:, 0]), 4),
                       "kl": round(float(a[:, 2].mean()), 4)}
    return out


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def voi_metrics(rows, scale):
    vr = [r for r in rows if "voi_target" in r and "voi" in r]
    if not vr:
        return {}
    x = np.array([v * scale for r in vr for v in r["voi"].values()])
    y = np.array([r["voi_target"][k] for r in vr for k in r["voi"]])
    return {"n_slots": len(x), "mse": round(float(((x - y) ** 2).mean()), 5), "spearman": round(spearman(x, y), 4)}


def grad_norms(model, terms, params):
    out = {}
    for name, loss in terms.items():
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        out[name] = float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)))
    return out


# ------------------------------------------------------------------ main loop
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train or fine-tune a Lavoir model.")
    ap.add_argument("--phase", choices=["decision", "joint", "temperature"], required=True)
    ap.add_argument("--data", nargs="+", required=True, help="training JSONL files (glob patterns allowed)")
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument("--encoder", default=None, help="decision from scratch: encoder path or Hub id")
    ap.add_argument("--init", default=None, help="checkpoint to start from (required for joint / temperature)")
    ap.add_argument("--voi_targets", default=None, help="joint: output of `python -m lavoir.targets`")
    ap.add_argument("--tiny", action="store_true", help="smoke test: small random encoder (needs --tokenizer)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer path or Hub id (default: the encoder's)")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--head_max_len", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=32, help="per GPU")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--warmup_steps", type=int, default=None,
                    help="head warm-up with a frozen encoder (default: 500 from scratch, 0 otherwise)")
    ap.add_argument("--lr_encoder", type=float, default=None, help="default: 2.5e-5 decision, 5e-6 joint")
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--sigma", type=float, nargs=2, default=[0.4, 0.1], help="GRPO noise, start and end")
    ap.add_argument("--w_rl", type=float, default=0.0, help="weight of the GRPO (RLCD) term; Laya's recipe is 1.0")
    ap.add_argument("--voi_weight", type=float, default=1.0, help="joint: lambda, weight of the VOI loss")
    ap.add_argument("--no_voi_cap", action="store_true", help="joint: do not bound VOI by Gini(p)")
    ap.add_argument("--calib_frac", type=float, default=0.05)
    ap.add_argument("--grad_log_every", type=int, default=200, help="joint: log RL / CE / VOI gradient norms")
    ap.add_argument("--max_steps", type=int, default=0, help="cap on steps (smoke tests)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gradient_checkpointing", action="store_true", help="trade speed for encoder memory")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.phase in ("joint", "temperature") and not a.init:
        ap.error("--phase %s needs --init" % a.phase)
    if a.phase == "joint" and not a.voi_targets:
        ap.error("--phase joint needs --voi_targets")
    if a.phase == "decision" and not a.init and not a.encoder and not a.tiny:
        ap.error("--phase decision needs --encoder (from scratch) or --init (fine-tuning)")
    if a.tiny and not (a.tokenizer or a.init):
        ap.error("--tiny needs --tokenizer")
    if a.warmup_steps is None:
        a.warmup_steps = 500 if (a.phase == "decision" and not a.init) else 0
    if a.lr_encoder is None:
        a.lr_encoder = 5e-6 if a.phase == "joint" else 2.5e-5
    return a


def main(argv=None):
    a = parse_args(argv)
    # DDP (torchrun): every GPU takes its share of the same shuffled batch list; evaluation, temperature
    # fitting, gradient-norm logging and saving happen on rank 0 only.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    lrank = int(os.environ.get("LOCAL_RANK", "0"))
    main_rank = rank == 0
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(lrank)
        device = torch.device("cuda", lrank)
    else:
        device = torch.device("cpu")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    amp = device.type == "cuda"
    log = open(os.path.join(a.out, "log.jsonl"), "a") if main_rank else None
    T0 = time.time()

    def barrier():
        if world > 1:
            dist.barrier()

    def L(**kw):
        if not main_rank:
            return
        kw["t"] = round(time.time() - T0, 1)
        log.write(json.dumps(kw, ensure_ascii=False) + "\n")
        log.flush()
        print(json.dumps(kw, ensure_ascii=False), flush=True)

    if a.init:
        model, tok, cfg = load_checkpoint(a.init)
        cfg = {k: v for k, v in cfg.items() if k != "training"}
    else:
        tok = load_tokenizer(a.tokenizer or a.encoder)
        cfg = {"encoder": "tiny" if a.tiny else a.encoder, "head_layers": 2, "n_act": 1,
               "max_len": a.max_len, "head_max_len": a.head_max_len}
        if a.tiny:
            model = LavoirModel(tiny_encoder(len(tok), tok.pad_token_id, d=64), n_act=cfg["n_act"])
        else:
            model = build_voi_model(cfg, a.encoder, pretrained=True)
    max_len, head_max_len = cfg.get("max_len", a.max_len), cfg.get("head_max_len", a.head_max_len)
    if a.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)

    exs = load_examples(a.data, "train")
    ids = {e["id"] for e in exs}
    if len(ids) != len(exs):
        raise ValueError("example ids must be unique")
    r = random.Random(a.seed)
    order = list(range(len(exs)))
    r.shuffle(order)
    n_cal = int(len(exs) * a.calib_frac)
    cal_ids = set()
    prev = os.path.join(a.init, "calib_ids.json") if a.init and os.path.isdir(a.init) else None
    if prev and os.path.exists(prev):             # same data as the init run: keep its held-out slice
        cal_ids = set(json.load(open(prev))) & ids
    if len(cal_ids) < min(n_cal, 10):
        cal_ids = {exs[i]["id"] for i in order[:n_cal]}
    if main_rank:
        json.dump(sorted(cal_ids), open(os.path.join(a.out, "calib_ids.json"), "w"))
    train = [e for e in exs if e["id"] not in cal_ids]
    calib = [e for e in exs if e["id"] in cal_ids]

    voi = {}
    if a.phase == "joint":
        for line in open(a.voi_targets):
            x = json.loads(line)
            voi[x["id"]] = x["voi"]
        # floor: if the targets are nearly constant (e.g. an untrained snapshot) the normalization must not explode
        sig_t = max(float(np.std([v for d in voi.values() for v in d.values()])), 1e-2)
        model.voi_scale.fill_(sig_t)
        model.voi_cap = not a.no_voi_cap
    else:
        sig_t = float(model.voi_scale)
    voi_wf = sorted({e.get("workflow") for e in exs if e.get("probes")})     # workflows with VOI data
    L(event="data", train=len(train), calib=len(calib), with_voi_target=sum(e["id"] in voi for e in train),
      sources={str(k): sum(e.get("workflow") == k for e in train) for k in sorted({e.get("workflow") for e in train}, key=str)},
      sigma_target=sig_t, device=str(device), world=world, global_batch=a.batch_size * world)

    ds = ExDataset(train, tok, max_len, head_max_len, voi, a.seed)
    ds_cal = ExDataset(calib, tok, max_len, head_max_len, voi, a.seed, shuffle=False)
    coll = make_collate(tok.pad_token_id)
    cal_loader = DataLoader(ds_cal, batch_sampler=length_batches(calib, a.batch_size, 0), collate_fn=coll,
                            num_workers=a.workers)

    raw = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # the VOI head (decision phase) and the frozen encoder (warm-up) leave unused parameters
        model = DDP(raw, device_ids=[lrank] if device.type == "cuda" else None, find_unused_parameters=True)
    enc_p = [p for n, p in raw.named_parameters() if n.startswith("encoder.")]
    head_p = [p for n, p in raw.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr_encoder}, {"params": head_p, "lr": a.lr_head}],
                            weight_decay=0.01)
    steps_per_epoch = len(length_batches(train, a.batch_size, 0)) // world
    warm = a.warmup_steps if a.phase == "decision" else 0
    total = 0 if a.phase == "temperature" else warm + int(steps_per_epoch * a.epochs)
    if a.max_steps:
        total = min(total, a.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))) * 0.98 + 0.02)
    shared = [p for n, p in raw.named_parameters() if n.startswith("head.")] + list(raw.encoder.parameters())[-10:]
    L(event="plan", total_steps=total, warmup_steps=warm, steps_per_epoch=steps_per_epoch)

    step, ep, last_b = 0, 0, None
    while step < total:
        ds.epoch = ep
        allb = length_batches(train, a.batch_size, a.seed + ep)
        mine = allb[rank::world][: len(allb) // world]       # every rank gets the same number of batches
        loader = DataLoader(ds, batch_sampler=mine, collate_fn=coll, num_workers=a.workers, persistent_workers=False)
        model.train()
        for b in loader:
            if b is None:                                   # every item dropped: reuse the previous batch so that
                if last_b is None:                          # all ranks keep the same step count
                    continue
                b = last_b
            last_b = b
            frozen = step < warm
            prog = min(1.0, max(0, step - warm) / max(1, total - warm))
            sigma = a.sigma[0] + (a.sigma[1] - a.sigma[0]) * prog
            inp = model_inputs(b, device)
            use_voi = a.phase == "joint" and "voi_mask" in b and b["voi_mask"].any().item()
            if not use_voi:
                inp["slot_pos"] = None
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(**inp, detach_encoder=frozen)
            d = decision_loss(out["logits"], b["target"].to(device), inp["marker_mask"], inp["qtype"], sigma, w_rl=a.w_rl)
            loss = d["loss"]
            lv = None
            if use_voi:
                lv = voi_loss(out["voi"], b["voi_target"].to(device), b["voi_mask"].to(device), sig_t)
                loss = loss + a.voi_weight * lv
            gn = None
            if main_rank and a.phase == "joint" and a.grad_log_every and step % a.grad_log_every == 0 and lv is not None:
                # separate forward pass outside the DDP wrapper: autograd.grad does not touch .grad or the sync
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    o2 = raw(**inp)
                d2 = decision_loss(o2["logits"], b["target"].to(device), inp["marker_mask"], inp["qtype"], sigma, w_rl=a.w_rl)
                lv2 = voi_loss(o2["voi"], b["voi_target"].to(device), b["voi_mask"].to(device), sig_t)
                gn = grad_norms(raw, {"rl": d2["rl"], "ce": d2["ce"], "voi": a.voi_weight * lv2}, shared)
                del o2, d2, lv2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if main_rank and (step % 50 == 0 or gn):
                L(event="step", step=step, epoch=ep, warmup=frozen, sigma=round(sigma, 3), loss=round(loss.item(), 4),
                  rl=round(d["rl"].item(), 4), ce=round(d["ce"].item(), 4), kl=round(d["kl"].item(), 4),
                  voi=None if lv is None else round(lv.item(), 4), grad_norm=gn,
                  lr=[round(g["lr"], 8) for g in opt.param_groups])
            step += 1
            if step >= total:
                break
        ep += 1
        if main_rank:
            rows = predict(raw, cal_loader, device, amp)
            L(event="epoch", epoch=ep, step=step, calib=metrics(rows, [1.0, 1.0, 1.0]), voi=voi_metrics(rows, sig_t))
        barrier()

    if not main_rank:
        barrier()
        return
    # temperature fitting on the held-out slice and final metrics (rank 0)
    model = raw
    rows = predict(model, cal_loader, device, amp)
    # One temperature for everything is fitted to over-confident one-hot general data and badly miscalibrates the
    # VOI workflows (whose soft targets are already matched at T ~ 1), so the two groups are fitted separately.
    # model.temperature = [VOI choice T, general score T, general noul T]: the VOI head reads the choice slot.
    ours = [r for r in rows if r["src"] in voi_wf]
    gen = [r for r in rows if r["src"] not in voi_wf]
    t_gen = [fit_one_temp([(r["logits"], r["target"]) for r in gen if r["qtype"] == q]) for q in range(3)]
    t_voi = [fit_one_temp([(r["logits"], r["target"]) for r in ours if r["qtype"] == q]) for q in range(3)]
    if sum(r["qtype"] == 0 for r in ours) >= 10:
        t_voi_choice = t_voi[0]
    elif cfg.get("temperature_voi"):                  # no VOI data now: keep the temperature the VOI head was trained with
        t_voi, t_voi_choice = list(cfg["temperature_voi"]), cfg["temperature_voi"][0]
    else:
        t_voi_choice = t_gen[0]
    temps = [t_voi_choice, t_gen[1], t_gen[2]]
    model.temperature.copy_(torch.tensor(temps))
    t_src = {}
    for src in sorted({r["src"] for r in gen}, key=str):
        rs = [r for r in gen if r["src"] == src]
        t_src[str(src)] = [fit_one_temp([(r["logits"], r["target"]) for r in rs if r["qtype"] == q])
                           if sum(r["qtype"] == q for r in rs) >= 10 else t_gen[q] for q in range(3)]

    def grp(rs, t):
        return metrics(rs, t)["_all"] if rs else {}
    L(event="temperature", T={QTYPE_NAMES[q]: round(t, 4) for q, t in enumerate(temps)},
      T_general={QTYPE_NAMES[q]: round(t, 4) for q, t in enumerate(t_gen)},
      T_voi={QTYPE_NAMES[q]: round(t, 4) for q, t in enumerate(t_voi)},
      T_by_source={k: [round(x, 3) for x in v] for k, v in t_src.items()},
      voi_T1=grp(ours, [1.0] * 3), voi_fit=grp(ours, temps), general_T1=grp(gen, [1.0] * 3), general_fit=grp(gen, t_gen),
      calib_T1=metrics(rows, [1.0, 1.0, 1.0]), voi=voi_metrics(rows, sig_t))
    with open(os.path.join(a.out, "calib_predictions.jsonl"), "w") as f:
        for r_ in rows:
            f.write(json.dumps(r_, ensure_ascii=False) + "\n")

    cfg.update(temperature_general=t_gen, temperature_voi=t_voi, temperature_by_source=t_src,
               voi_workflows=[w for w in voi_wf if w is not None] or cfg.get("voi_workflows", []),
               training={"phase": a.phase, "init": a.init, "voi_targets": a.voi_targets, "args": vars(a)})
    save_checkpoint(model, tok, cfg, a.out)
    L(event="done", out=a.out)
    barrier()


if __name__ == "__main__":
    main()
