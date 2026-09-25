"""Inference API.

    from lavoir import Lavoir

    model = Lavoir.from_pretrained("path/to/checkpoint")       # or a Hugging Face Hub repo id
    pred = model.predict(state, question, slots)
    pred.probabilities   # {option: probability}
    pred.voi             # {slot: expected gain in p(correct option) if its question is asked}
    step = model.next_action(state, question, slots)          # ask / decide / handoff
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .checkpoint import calibration, load_checkpoint, temperature_tensor
from .collate import collate_voi, model_inputs
from .items import make_item
from .policy import ASK, Step, decide, run_dialogue, slot_spec


@dataclass
class Prediction:
    probabilities: Dict[str, float]
    voi: Dict[str, float] = field(default_factory=dict)

    @property
    def choice(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def confidence(self) -> float:
        return max(self.probabilities.values())

    @property
    def best_question(self) -> Optional[str]:
        return max(self.voi, key=self.voi.get) if self.voi else None


class Lavoir:
    """A decision model that also estimates, for every candidate question, how much asking it would help.

    state:     a string, a dict, or a conversation [{"role": "user" | "system", "text": ...}, ...]
               (the model was trained with "user" for the customer and "system" for the assistant's questions)
    question:  {"type": "choice" | "noul" | "score", "instructions": ..., "criteria": ...} (Laya format)
    slots:     {name: description} or {name: {"description": ..., "question": ...}}: information that is still
               missing. Leave out slots that were already asked.
    """

    def __init__(self, model, tokenizer, config: Dict, device=None, batch_size: int = 32):
        self.model, self.tokenizer, self.config = model.eval(), tokenizer, config
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.max_len = config.get("max_len", 1024)
        self.head_max_len = config.get("head_max_len", 384)
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(cls, path_or_repo: str, device=None, revision: Optional[str] = None, **kw) -> "Lavoir":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tok, cfg = load_checkpoint(path_or_repo, device, revision)
        return cls(model, tok, cfg, device, **kw)

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict_batch(self, requests: Sequence[Tuple], calibration_source: Optional[str] = None) -> List[Prediction]:
        """requests: [(state, question, slots or None), ...]."""
        items = []
        for state, question, slots in requests:
            desc = {s: v["description"] for s, v in slot_spec(slots or {}).items()}
            it = make_item(self.tokenizer, {"state": state, "question": question, "slots": desc},
                           self.max_len, self.head_max_len)
            if it is None:
                raise ValueError("the options or slots do not fit in max_len=%d tokens" % self.max_len)
            items.append(it)
        order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
        out: List[Optional[Prediction]] = [None] * len(items)
        use_amp = self.device.type == "cuda"
        for k in range(0, len(order), self.batch_size):
            idx = order[k:k + self.batch_size]
            b = collate_voi([items[i] for i in idx], self.tokenizer.pad_token_id)
            inp = model_inputs(b, self.device)
            if not b["slot_mask"].any():
                inp["slot_pos"] = None
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                o = self.model(**inp)
            v = (o["voi"].float() * self.model.voi_scale).cpu() if o["voi"] is not None else None
            for j, i in enumerate(idx):
                it = items[i]
                temps = calibration(self.config, calibration_source, bool(it["slot_names"]))
                T = temperature_tensor(self.model, temps)[it["qtype"]].clamp_min(1e-3)
                n = len(it["option_keys"])
                p = torch.softmax(o["logits"][j, :n].float() / T, -1).cpu().tolist()
                voi = dict(zip(it["slot_names"], v[j, :len(it["slot_names"])].tolist())) \
                    if v is not None and it["slot_names"] else {}
                out[i] = Prediction(dict(zip(it["option_keys"], p)), voi)
        return out

    def predict(self, state, question: Dict, slots: Optional[Dict] = None,
                calibration_source: Optional[str] = None) -> Prediction:
        """calibration_source: None picks the VOI-workflow temperature when slots are given and the
        general-data temperature otherwise; "voi", "general" or a training source name force one."""
        return self.predict_batch([(state, question, slots)], calibration_source)[0]

    def __call__(self, state, question: Dict, slots: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Predictor interface used by `lavoir.policy.run_dialogue`."""
        pred = self.predict(state, question, slots)
        return pred.probabilities, pred.voi

    # ------------------------------------------------------------------ acting
    def next_action(self, state, question: Dict, slots: Dict, asked: Iterable[str] = (),
                    ask_threshold: float = 0.05, handoff_threshold: Optional[float] = None,
                    max_questions: int = 3) -> Step:
        """One step of the rule. `slots` should hold the slots not asked yet; `asked` lists those asked."""
        asked = set(asked)
        spec = {s: v for s, v in slot_spec(slots).items() if s not in asked}
        pred = self.predict(state, question, spec)
        st = decide(pred.probabilities, pred.voi, asked, len(asked), ask_threshold, handoff_threshold,
                    max_questions)
        if st.action == ASK:
            st.question = spec[st.slot]["question"] or st.slot
        return st

    def run_dialogue(self, state, question: Dict, slots: Dict, answer: Callable[[str, str], str],
                     ask_threshold: float = 0.05, handoff_threshold: Optional[float] = None,
                     max_questions: int = 3):
        """Asks questions through `answer(slot, question_text) -> reply` until it decides or hands off."""
        return run_dialogue(self, state, question, slots, answer, ask_threshold, handoff_threshold, max_questions)
