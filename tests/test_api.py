import random

import pytest
import torch

from lavoir import ASK, DECIDE, Lavoir, save_checkpoint
from lavoir.collate import collate_voi, model_inputs
from lavoir.items import make_item
from lavoir.model import LavoirModel, tiny_encoder

from conftest import rand_example


@pytest.fixture
def ckpt(tok, tmp_path):
    torch.manual_seed(0)
    m = LavoirModel(tiny_encoder(len(tok), tok.pad_token_id), n_act=1, voi_cap=True).eval()
    with torch.no_grad():
        m.scorer[-1].weight.mul_(300.0)     # a random tiny model gives near-equal logits; spread them out
        m.voi_scale.fill_(0.2)
        m.temperature.copy_(torch.tensor([1.1, 3.0, 2.0]))
    cfg = {"encoder": "tiny", "head_layers": 2, "n_act": 1, "max_len": 1024, "head_max_len": 384,
           "temperature_general": [4.0, 3.0, 2.0], "temperature_voi": [1.1, 1.0, 1.0],
           "temperature_by_source": {"src_a": [2.5, 3.0, 2.0]}}
    save_checkpoint(m, tok, cfg, str(tmp_path / "ckpt"))
    return m, str(tmp_path / "ckpt")


def test_roundtrip_matches_model(tok, ckpt):
    m, path = ckpt
    api = Lavoir.from_pretrained(path, device="cpu")
    assert api.model.voi_cap and abs(float(api.model.voi_scale) - 0.2) < 1e-7
    ex = rand_example(random.Random(0), n_opt=4, n_slot=3)
    pred = api.predict(ex["state"], ex["question"], ex["slots"])
    it = make_item(tok, {"state": ex["state"], "question": ex["question"], "slots": ex["slots"]})
    with torch.no_grad():
        out = m(**model_inputs(collate_voi([it], tok.pad_token_id), "cpu"))
    p = torch.softmax(out["logits"][0] / 1.1, -1).tolist()
    v = (out["voi"][0] * 0.2).tolist()
    assert pred.probabilities == pytest.approx(dict(zip(it["option_keys"], p)), abs=1e-6)
    assert pred.voi == pytest.approx(dict(zip(it["slot_names"], v)), abs=1e-6)
    assert pred.choice == max(pred.probabilities, key=pred.probabilities.get)


def test_calibration_sources(ckpt):
    _, path = ckpt
    api = Lavoir.from_pretrained(path, device="cpu")
    ex = rand_example(random.Random(1), n_opt=3, n_slot=2)
    with_slots = api.predict(ex["state"], ex["question"], ex["slots"])
    no_slots = api.predict(ex["state"], ex["question"])
    forced = api.predict(ex["state"], ex["question"], calibration_source="voi")
    by_src = api.predict(ex["state"], ex["question"], calibration_source="src_a")
    assert no_slots.voi == {} and with_slots.voi
    # the general temperature (4.0) is flatter than the VOI one (1.1)
    assert no_slots.confidence < forced.confidence
    assert no_slots.confidence < by_src.confidence < forced.confidence
    with pytest.raises(KeyError):
        api.predict(ex["state"], ex["question"], calibration_source="nope")


def test_next_action_and_batch(ckpt):
    _, path = ckpt
    api = Lavoir.from_pretrained(path, device="cpu")
    ex = rand_example(random.Random(2), n_opt=3, n_slot=3)
    slots = {s: {"description": d, "question": "Tell me about %s?" % s} for s, d in ex["slots"].items()}
    st = api.next_action(ex["state"], ex["question"], slots, ask_threshold=-1.0)
    assert st.action == ASK and st.question == "Tell me about %s?" % st.slot
    st2 = api.next_action(ex["state"], ex["question"], slots, ask_threshold=2.0)
    assert st2.action == DECIDE
    st3 = api.next_action(ex["state"], ex["question"], slots, asked=list(slots), ask_threshold=-1.0)
    assert st3.action == DECIDE                                   # nothing left to ask
    reqs = [(ex["state"], ex["question"], ex["slots"]), (ex["state"], ex["question"], None)]
    batch = api.predict_batch(reqs)
    single = [api.predict(*r) for r in reqs]
    # padding changes the float summation order; the fixture's 300x scorer amplifies it (transformers 5: ~5e-6)
    for a, b in zip(batch, single):
        assert a.probabilities == pytest.approx(b.probabilities, abs=1e-5)
        assert a.voi == pytest.approx(b.voi, abs=1e-5)


def test_checkpoint_tokenizer_portable(tok, ckpt):
    """Checkpoints must load on transformers 4.x and 5.x: the saved tokenizer class is one both resolve, and a
    class name this version does not know (5.x writes "TokenizersBackend") falls back to the same tokenizer.json."""
    import json
    import os
    m, path = ckpt
    tc = os.path.join(path, "tokenizer", "tokenizer_config.json")
    with open(tc) as f:
        assert json.load(f)["tokenizer_class"] == "PreTrainedTokenizerFast"
    ex = rand_example(random.Random(1), n_opt=3, n_slot=2)
    ref = Lavoir.from_pretrained(path, device="cpu").predict(ex["state"], ex["question"], ex["slots"])
    with open(tc) as f:
        cfg = json.load(f)
    cfg["tokenizer_class"] = "NoSuchTokenizerClass"
    with open(tc, "w") as f:
        json.dump(cfg, f)
    api = Lavoir.from_pretrained(path, device="cpu")
    pred = api.predict(ex["state"], ex["question"], ex["slots"])
    assert api.tokenizer("merhaba dünya")["input_ids"] == tok("merhaba dünya")["input_ids"]
    assert pred.probabilities == pytest.approx(ref.probabilities, abs=1e-6)
    assert pred.voi == pytest.approx(ref.voi, abs=1e-6)
