import os
import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

# Tokenizer used by the unit tests; any ModernBERT tokenizer works. Set LAVOIR_TOKENIZER to a local copy to
# run offline. LAVOIR_ENCODER (a local ModernBERT-large directory) enables the slow full-size test.
TOKENIZER = os.environ.get("LAVOIR_TOKENIZER", "answerdotai/ModernBERT-large")
ENCODER_DIR = os.environ.get("LAVOIR_ENCODER")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the full ModernBERT-large weights (LAVOIR_ENCODER)")


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER)


class BagEncoder(nn.Module):
    """Position-free encoder: the output depends on token identity only (for the permutation test)."""

    def __init__(self, vocab: int, d: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(hidden_size=d)
        self.emb = nn.Embedding(vocab, d)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


@pytest.fixture
def rng():
    return random.Random(1234)


WORDS = ("account refund invoice password delete data contract health payment login error charge "
         "report manager shipping delay broken screen access request complaint policy claim damage").split()


def rand_text(r: random.Random, n: int) -> str:
    return " ".join(r.choice(WORDS) for _ in range(n))


def rand_example(r: random.Random, n_opt=None, n_slot=None, with_voi=True, idx=0):
    n_opt = n_opt or r.randint(3, 5)
    n_slot = r.randint(2, 4) if n_slot is None else n_slot
    units = {"unit%d" % i: rand_text(r, 5) for i in range(n_opt)}
    slots = {"slot%d" % j: rand_text(r, 4) for j in range(n_slot)}
    w = [r.random() ** 3 for _ in units]
    s = sum(w)
    ex = {"id": "ex%d" % idx, "workflow": "wf",
          "state": [{"role": "user", "text": rand_text(r, r.randint(10, 30))}],
          "question": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": units},
          "target": {k: v / s for k, v in zip(units, w)}, "slots": slots,
          "gold": r.choice(list(units))}
    if with_voi:
        ex["voi"] = {k: r.uniform(-0.2, 0.6) for k in slots}
    ex["probes"] = {k: {"q": "What about %s?" % k, "a": rand_text(r, 4)} for k in slots}
    return ex
