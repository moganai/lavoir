"""Interactive demo: you play the customer, the model asks, decides or hands off.

    python examples/chat_demo.py --model path/to/lavoir-model --workflow examples/workflows/it_helpdesk.json

A workflow file lists the options and the questions the assistant may ask:
    {"instruction": "...", "options": {"team": "description", ...},
     "slots": {"slot": {"description": "...", "question": "..."}, ...}}
Empty line: new conversation. q: quit.
"""
import argparse
import json

from lavoir import ASK, HANDOFF, Lavoir


def bar(x, width=24):
    n = max(0, min(width, round(x * width)))
    return "#" * n + "." * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--ask_threshold", type=float, default=0.05, help="ask while the best question's VOI exceeds this")
    ap.add_argument("--handoff_threshold", type=float, default=None,
                    help="hand off when 1 - max p exceeds this after the questions (default: never)")
    ap.add_argument("--max_questions", type=int, default=3)
    a = ap.parse_args()
    model = Lavoir.from_pretrained(a.model)
    wf = json.load(open(a.workflow))
    question = {"type": "choice", "instructions": wf["instruction"], "criteria": wf["options"]}
    print("options: %s\n" % ", ".join(wf["options"]))
    while True:
        msg = input("customer> ").strip()
        if msg == "q":
            return
        if not msg:
            continue
        state, asked = [{"role": "user", "text": msg}], []
        while True:
            st = model.next_action(state, question, wf["slots"], asked, a.ask_threshold, a.handoff_threshold,
                                   a.max_questions)
            for k, v in sorted(st.p.items(), key=lambda t: -t[1]):
                print("   %-22s %s %.3f" % (k, bar(v), v))
            if st.voi:
                print("   VOI: " + ", ".join("%s %.3f" % (k, v) for k, v in sorted(st.voi.items(), key=lambda t: -t[1])))
            if st.action == ASK:
                reply = input("assistant> %s\ncustomer> " % st.question).strip()
                if reply == "q":
                    return
                state += [{"role": "system", "text": st.question}, {"role": "user", "text": reply}]
                asked.append(st.slot)
                continue
            if st.action == HANDOFF:
                print("-> hand off to a human\n")
            else:
                print("-> route to %s (p = %.3f, %d question(s))\n" % (st.choice, st.p[st.choice], len(asked)))
            break


if __name__ == "__main__":
    main()
