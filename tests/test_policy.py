from lavoir.policy import ASK, DECIDE, HANDOFF, decide, run_dialogue

SLOTS = {s: {"description": "d " + s, "question": "Q " + s + "?"} for s in ("a", "b", "c")}
Q = {"type": "choice", "instructions": "Which?", "criteria": {"x": "x", "y": "y"}}


def test_never_reasks():
    calls = []

    def predict(state, q, slots):
        calls.append(set(slots))
        return {"x": 0.9, "y": 0.1}, {s: {"a": 1.0, "b": 0.5, "c": 0.3}[s] for s in slots}

    final, trace, _ = run_dialogue(predict, "msg", Q, SLOTS, lambda s, qt: "no idea", 0.1, 0.5, max_questions=3)
    asked = [t.slot for t in trace if t.action == ASK]
    assert asked == ["a", "b", "c"]
    assert [t.question for t in trace if t.action == ASK] == ["Q a?", "Q b?", "Q c?"]
    assert calls[1] == {"b", "c"} and calls[2] == {"c"}     # an asked slot leaves the list
    assert final.action == DECIDE and final.choice == "x"


def test_decide_ignores_asked_even_if_predictor_returns_it():
    st = decide({"x": 0.9, "y": 0.1}, {"a": 5.0, "b": 0.2}, asked={"a"}, n_asked=1, ask_threshold=0.1,
                handoff_threshold=0.5)
    assert st.action == ASK and st.slot == "b"


def test_max_questions_and_handoff():
    def predict(state, q, slots):
        return {"x": 0.55, "y": 0.45}, {s: 1.0 for s in slots}

    final, trace, state = run_dialogue(predict, "msg", Q, SLOTS, lambda s, qt: "ok", 0.1, 0.3, max_questions=2)
    assert sum(t.action == ASK for t in trace) == 2
    assert final.action == HANDOFF
    assert [t["role"] for t in state] == ["user", "system", "user", "system", "user"]


def test_no_handoff_threshold_means_decide():
    st = decide({"x": 0.4, "y": 0.35, "z": 0.25}, {}, asked=set(), n_asked=0, ask_threshold=0.05)
    assert st.action == DECIDE and st.choice == "x"


def test_plain_description_slots():
    def predict(state, q, slots):
        assert all(isinstance(v, str) for v in slots.values())
        return {"x": 0.5, "y": 0.5}, {s: 0.3 for s in slots}

    final, trace, _ = run_dialogue(predict, "msg", Q, {"a": "desc a"}, lambda s, qt: "ok", 0.1, None, 1)
    assert trace[0].action == ASK and trace[0].question == "a"     # no question text given: the slot name
    assert final.action == DECIDE
