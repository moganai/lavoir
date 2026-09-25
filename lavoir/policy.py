"""The ask / decide / hand-off rule.

    k = argmax(voi) over the slots not asked yet
    voi[k] > ask_threshold and questions < max_questions   -> ask the question of slot k
    1 - max(p) > handoff_threshold                           -> hand off to a human
    otherwise                                                -> decide argmax(p)

A predictor maps (state, question, slots {name: description}) to
(p {option: probability}, voi {slot: expected gain in p(correct option)}).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .items import append_turns

ASK, DECIDE, HANDOFF = "ask", "decide", "handoff"


@dataclass
class Step:
    action: str
    slot: Optional[str] = None
    question: Optional[str] = None
    choice: Optional[str] = None
    p: Dict[str, float] = field(default_factory=dict)
    voi: Dict[str, float] = field(default_factory=dict)


def decide(p: Dict[str, float], voi: Dict[str, float], asked, n_asked: int, ask_threshold: float,
           handoff_threshold: Optional[float] = None, max_questions: int = 3) -> Step:
    cand = {k: v for k, v in voi.items() if k not in asked}
    if cand and n_asked < max_questions:
        k = max(cand, key=cand.get)
        if cand[k] > ask_threshold:
            return Step(ASK, slot=k, p=p, voi=voi)
    best = max(p, key=p.get)
    if handoff_threshold is not None and 1.0 - p[best] > handoff_threshold:
        return Step(HANDOFF, p=p, voi=voi)
    return Step(DECIDE, choice=best, p=p, voi=voi)


def slot_spec(slots: Dict) -> Dict[str, Dict[str, Optional[str]]]:
    """Normalizes {name: description} or {name: {"description"|"desc", "question"}}."""
    out = {}
    for s, v in slots.items():
        if isinstance(v, dict):
            out[s] = {"description": v.get("description", v.get("desc", "")), "question": v.get("question")}
        else:
            out[s] = {"description": str(v), "question": None}
    return out


def run_dialogue(predict: Callable, state, question: Dict, slots: Dict, answer: Callable[[str, str], str],
                 ask_threshold: float = 0.05, handoff_threshold: Optional[float] = None,
                 max_questions: int = 3) -> Tuple[Step, List[Step], list]:
    """Runs the rule until it decides or hands off.

    slots: {name: {"description", "question"}}. answer(slot, question_text) returns the user's reply.
    An asked slot leaves the list, even if the reply was "I don't know".
    Returns (final step, all steps, final state).
    """
    spec = slot_spec(slots)
    state = append_turns(state, [])
    asked: set = set()
    trace: List[Step] = []
    while True:
        remaining = {s: v["description"] for s, v in spec.items() if s not in asked}
        p, voi = predict(state, question, remaining)
        st = decide(p, voi, asked, len(asked), ask_threshold, handoff_threshold, max_questions)
        trace.append(st)
        if st.action != ASK:
            return st, trace, state
        st.question = spec[st.slot]["question"] or st.slot
        state = append_turns(state, [{"role": "system", "text": st.question},
                                     {"role": "user", "text": answer(st.slot, st.question)}])
        asked.add(st.slot)
