"""End-to-end smoke test of the CLI with a tiny random encoder on CPU:
decision (from scratch) -> VOI targets -> joint -> temperature -> decision fine-tuning -> evaluation."""
import json
import random
import sys

from lavoir import Lavoir
from lavoir import evaluate as evaluate_mod
from lavoir import targets as targets_mod
from lavoir import train as train_mod

from conftest import TOKENIZER, rand_example


def _write_data(path, n=40, seed=0):
    r = random.Random(seed)
    with open(path, "w") as f:
        for i in range(n):
            ex = rand_example(r, idx=i)
            ex.pop("voi")
            if i % 2:                                  # half general single-turn data: no slots, no probes
                ex.update(workflow="general", slots={}, probes=None)
                del ex["probes"]
            f.write(json.dumps(ex) + "\n")
            t = dict(ex, id="test%d" % i, split="test")
            f.write(json.dumps(t) + "\n")


def _run(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"] + argv)
    mod.main()


def test_full_pipeline(tmp_path, monkeypatch):
    data = str(tmp_path / "data.jsonl")
    _write_data(data)
    common = ["--data", data, "--batch_size", "4", "--max_steps", "3", "--workers", "0", "--calib_frac", "0.25"]
    dec = str(tmp_path / "decision")
    train_mod.main(["--phase", "decision", "--tiny", "--tokenizer", TOKENIZER, "--out", dec,
                    "--warmup_steps", "1"] + common)
    tgt = str(tmp_path / "targets.jsonl")
    _run(targets_mod, ["--checkpoint", dec, "--data", data, "--out", tgt, "--batch_size", "8"], monkeypatch)
    rows = [json.loads(l) for l in open(tgt)]
    assert len(rows) == 20 and all(r["voi"] for r in rows)

    joint = str(tmp_path / "joint")
    train_mod.main(["--phase", "joint", "--init", dec, "--voi_targets", tgt, "--out", joint] + common)
    cfg = json.load(open(tmp_path / "joint" / "config.json"))
    assert cfg["voi_cap"] is True and cfg["voi_workflows"] == ["wf"]
    assert cfg["training"]["phase"] == "joint" and len(cfg["temperature"]) == 3

    temp = str(tmp_path / "temperature")
    train_mod.main(["--phase", "temperature", "--init", joint, "--out", temp] + common)

    # fine-tuning on decision-only data keeps the VOI head's temperature
    general = str(tmp_path / "general.jsonl")
    with open(general, "w") as f:
        for l in open(data):
            e = json.loads(l)
            if e["workflow"] == "general" and e.get("split", "train") == "train":
                f.write(json.dumps(e) + "\n")
    ft = str(tmp_path / "finetuned")
    train_mod.main(["--phase", "decision", "--init", temp, "--out", ft, "--data", general, "--batch_size", "4",
                    "--max_steps", "2", "--workers", "0", "--calib_frac", "0.25"])
    cfg_t, cfg_f = (json.load(open(tmp_path / d / "config.json")) for d in ("temperature", "finetuned"))
    assert cfg_f["temperature"][0] == cfg_t["temperature"][0]
    assert cfg_f["voi_workflows"] == ["wf"] and cfg_f["voi_cap"] is True

    api = Lavoir.from_pretrained(ft, device="cpu")
    ex = rand_example(random.Random(9), n_opt=3, n_slot=2)
    pred = api.predict(ex["state"], ex["question"], ex["slots"])
    assert abs(sum(pred.probabilities.values()) - 1) < 1e-5 and set(pred.voi) == set(ex["slots"])

    out = str(tmp_path / "eval")
    _run(evaluate_mod, ["--checkpoint", ft, "--data", data, "--out", out], monkeypatch)
    summ = json.load(open(tmp_path / "eval" / "summary.json"))
    assert summ["decision"]["_all"]["n"] == 40 and summ["voi"]["n_slots"] > 0
