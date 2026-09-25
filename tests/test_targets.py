import random

import torch

from lavoir.collate import collate_voi, model_inputs
from lavoir.items import append_turns, make_item
from lavoir.model import LavoirModel, tiny_encoder
from lavoir.targets import compute_voi_targets

from conftest import rand_example


def test_voi_target_is_pk_minus_p0(tok):
    torch.manual_seed(0)
    m = LavoirModel(tiny_encoder(len(tok), tok.pad_token_id)).eval()
    with torch.no_grad():
        m.temperature.fill_(1.7)
    r = random.Random(5)
    exs = [rand_example(r, idx=i) for i in range(6)]
    res = compute_voi_targets(m, tok, exs, torch.device("cpu"), batch_size=4)
    assert len(res) == 6
    for ex, rr in zip(exs, res):
        assert set(rr["voi"]) == set(ex["slots"])
        for k, v in rr["voi"].items():
            assert abs(v - (rr["pk_gold"][k] - rr["p0_gold"])) < 1e-9
        assert abs(sum(rr["p0"]) - 1) < 1e-5

    # independent recomputation of pk for one slot
    ex, rr = exs[0], res[0]
    k = next(iter(ex["slots"]))
    st = append_turns(ex["state"], [{"role": "system", "text": ex["probes"][k]["q"]},
                                    {"role": "user", "text": ex["probes"][k]["a"]}])
    it = make_item(tok, ex, state=st, slots={s: d for s, d in ex["slots"].items() if s != k})
    with torch.no_grad():
        out = m(**{**model_inputs(collate_voi([it], tok.pad_token_id), "cpu"), "slot_pos": None})
    p = torch.softmax(out["logits"][0] / 1.7, -1)
    assert abs(float(p[it["gold_idx"]]) - rr["pk_gold"][k]) < 1e-5
