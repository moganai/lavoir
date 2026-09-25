"""Quickstart: one routing decision, the value of each candidate question, and a full two-question dialogue.

    python examples/quickstart.py --model path/to/lavoir-model
"""
import argparse
import json
import os

from lavoir import Lavoir

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint directory or Hugging Face Hub repo id")
    a = ap.parse_args()
    model = Lavoir.from_pretrained(a.model)

    wf = json.load(open(os.path.join(HERE, "workflows", "ecommerce_returns.json")))
    question = {"type": "choice", "instructions": wf["instruction"], "criteria": wf["options"]}
    state = [{"role": "user", "text": "hi, i need help with my recent purchase. i would like a replacement for the "
                                      "item. can you help me with that? thanks"}]

    # 1) One forward pass: probabilities over the teams and the value of every question.
    pred = model.predict(state, question, wf["slots"])
    print("probabilities:", {k: round(v, 3) for k, v in sorted(pred.probabilities.items(), key=lambda t: -t[1])})
    print("VOI:          ", {k: round(v, 3) for k, v in sorted(pred.voi.items(), key=lambda t: -t[1])})

    # 2) The decision rule, run until it decides. Answers are scripted here; in a real chat they come from the user.
    replies = {"seller": "The item was sold by our store.", "problem": "The item arrived damaged."}
    final, trace, _ = model.run_dialogue(state, question, wf["slots"],
                                         answer=lambda slot, q: replies.get(slot, "I'm not sure."),
                                         ask_threshold=0.05)
    for step in trace:
        if step.action == "ask":
            print("ask   %-13s VOI %.3f  %s  ->  %s" % (step.slot, step.voi[step.slot], step.question, replies.get(step.slot)))
    print("%s: %s (p = %.3f)" % (final.action, final.choice, final.p[final.choice]) if final.choice else final.action)


if __name__ == "__main__":
    main()
