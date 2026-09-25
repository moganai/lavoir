---
language: en
license: cc-by-nc-4.0
base_model: answerdotai/ModernBERT-large
pipeline_tag: text-classification
tags:
  - lavoir
  - laya
  - value-of-information
  - clarifying-questions
  - calibrated-decisions
  - routing
  - customer-support
  - guardrails
---

<p align="center">
  <img src="banner.png" alt="Lavoir" width="600"/>
</p>

<p align="center">
  <a href="https://github.com/moganai/lavoir"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"/></a>
  <a href="https://moganai.github.io/lavoir-web"><img src="https://img.shields.io/badge/🌐_Blog-MoganAI-2E7D5B?style=flat-square" alt="Blog"/></a>
  <a href="https://buymeacoffee.com/moganai"><img src="https://img.shields.io/badge/Buy_Me_a_Coffee-FFDD00?style=flat-square&logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"/></a>
</p>

# Lavoir

**A decision model that knows which question to ask.**

Lavoir reads a conversation, a typed question (which team? yes or no? how severe?) and a list of things it could
still ask. In one forward pass it returns calibrated probabilities for the options and, for every candidate
question, its **value of information (VOI)**: how much the probability of the right answer is expected to rise if
that question is asked. One rule turns this into behaviour: ask the most valuable question while it is worth it,
then decide or hand off to a human.

```text
customer   hi, i need help with my recent purchase. i would like a replacement for the item.
           can you help me with that? thanks

           returns_desk 0.350   logistics 0.297   marketplace_support 0.287   warranty_service 0.066
VOI        seller 0.197   problem 0.171   delivery_age 0.036   wants 0.003   customer_name 0.001

Lavoir     Was the item sold by us directly or by a seller on our marketplace?        (VOI 0.197)
customer   The item was sold by our store.
Lavoir     What is the problem with your order?                                       (VOI 0.385)
customer   The item arrived damaged.

decision   logistics (p = 1.000)
```

From the first message alone the best guess would have been `returns_desk`, which is wrong. After the first
answer two teams are still close, so the second question is worth more than the first was. The customer's name
never gets asked: it cannot change the decision.

| | |
|---|---|
| Asks like an oracle | Question-asking curve matches an oracle that knows the exact routing rules (AUC .799 vs .797) |
| Asks only when it matters | asks in 6% of real SGD conversations; where it does not ask, its decision is right 96% of the time |
| Still a strong classifier | ahead of Laya's English checkpoint on 7 of 12 single-turn benchmarks, level on spam and phishing |
| Fast | one forward pass, 31 ms median on a GH200; no text generation |

## Quick start

```bash
pip install -e .   # from the Lavoir code repository (installs laya, torch, transformers)
```

```python
from lavoir import Lavoir

model = Lavoir.from_pretrained("moganai/lavoir")     # or a local copy of this repository

question = {"type": "choice", "instructions": "Which team should handle this order problem?",
            "criteria": {"returns_desk": "returns desk: standard returns and exchanges",
                         "logistics": "logistics: replacements for damaged or wrong deliveries",
                         "marketplace_support": "marketplace support: orders from third-party sellers",
                         "warranty_service": "warranty service: products that stopped working after the return window"}}
slots = {"seller": {"description": "who sold the item",
                    "question": "Was the item sold by us directly or by a seller on our marketplace?"},
         "problem": {"description": "what is wrong with the order",
                     "question": "What is the problem with your order?"},
         "delivery_age": {"description": "how long ago the item was delivered",
                          "question": "When was it delivered: within the last 30 days or earlier?"}}
state = [{"role": "user", "text": "hi, i need help with my recent purchase. i would like a replacement for the item."}]

pred = model.predict(state, question, slots)
pred.probabilities        # {"returns_desk": 0.35, "logistics": 0.30, ...}
pred.voi                  # {"seller": 0.20, "problem": 0.17, "delivery_age": 0.04}

step = model.next_action(state, question, slots, ask_threshold=0.05)
step.action, step.question   # ("ask", "Was the item sold by us directly or by a seller on our marketplace?")
```

- `state`: a string, a dict of named fields, or a list of turns (`"user"` for the customer, `"system"` for the
  assistant's questions).
- `question`: Laya's typed format: `choice` with `{option: description}`, `noul` (yes/no), `score` (ordered
  levels). It also works without `slots`, as a plain calibrated classifier.
- `slots`: the information still missing. Remove a slot once it has been asked.
- `ask_threshold=0.05` asks while a question is expected to raise p(correct) by at least 5 points. Add
  `handoff_threshold=0.3` to hand off when the best option stays below 70%.

## How it works

```
[CLS] choice question: <instruction> [SEP]
[MASK] <option 1>: <description>  [MASK] <option 2>: ...  [SEP]
missing information: [MASK] <slot 1>: <description>  [MASK] <slot 2>: ...  [SEP]
<conversation> [SEP]
```

- **Decision head** (from Laya): each option marker is scored; a softmax with a fitted temperature gives the
  probabilities. Without slots, the input is exactly Laya's format and the model behaves as a Laya-style
  classifier.
- **VOI head**: each slot marker's hidden state, together with summary features of the decision distribution,
  goes through a small MLP. Its output is bounded by the decision's Gini impurity,
  `VOI = (1 - Σp²) · sigmoid(h)`. A calibrated model cannot expect to gain more than that, so a confident model
  structurally does not ask.
- **Training target** for slot *k*: `p_k(correct) − p_0(correct)`, the change in the model's probability for
  the correct option after the question of slot *k* and one realistic answer are appended. The head learns the
  expectation of this quantity over answers.
- **Rule**: ask `argmax VOI` while it exceeds `ask_threshold`; then decide `argmax p`, or hand off if
  `1 − max p > handoff_threshold`.

## Evaluation

All numbers are on data that was not used for training.

### Asking the right question

8 customer-service workflows (banking, e-commerce returns, HR, insurance claims, IT help desk, privacy requests,
telecom, travel changes) with held-out conversations, and 4 workflows never seen in training (clinic routing,
parcel delivery, SaaS support, student affairs). Every conversation has a known correct team and every question a
realistic answer, so any policy can be replayed. The **oracle** knows each workflow's exact rules and priors and
computes the true expected VOI; it is the ceiling for any question-asking policy.

![Accuracy vs. questions asked](assets/questions_vs_accuracy.png)

*Each line is one strategy for deciding when and what to ask, replayed on the same conversations. The x-axis is
how many questions it asks per conversation on average, the y-axis how many conversations end with the correct
team. Lavoir (orange) climbs as fast as the oracle that knows the true rules on seen workflows; on unseen
workflows it still beats both baselines.*

| policy | AUC, seen workflows | accuracy at ≤ 0.5 questions, seen | AUC, unseen workflows | accuracy at ≤ 0.5 questions, unseen |
|---|---|---|---|---|
| never ask | .610 | .610 | .496 | .496 |
| random question when unsure | .760 | .663 | .572 | .529 |
| conformal set > 1, ask the VOI question | .789 | .610 | .592 | .496 |
| **Lavoir** | **.799** | **.751** | **.612** | **.581** |
| oracle VOI (exact) | .797 | .760 | .649 | |

AUC: normalized area under the accuracy-vs-questions curve from 0 to 2 questions per conversation.

| on the 8 training workflows | |
|---|---|
| first-message accuracy vs. Bayes ceiling | .648 vs .659 (ECE .016): no leakage, no overconfidence |
| VOI vs. oracle VOI | Spearman .85; same best question 94% of the time |
| `ask_threshold = 0.05` | 1.06 questions per conversation, accuracy .610 → .885, 0.3% of questions redundant |
| `ask_threshold = 0.02` + `handoff_threshold = 0.3` | 17% handed off, accuracy .970 on the rest |

On unseen workflows Lavoir still beats every baseline, but the gap to the oracle is larger and the decision itself
is weaker (first-message accuracy .514 vs a ceiling of .667). Fine-tuning on your own workflow closes most of this
gap.

### Real conversations (zero-shot)

Neither dataset was used for training. Candidate questions were generic (for ABCD) or the service's own slots
(for SGD).

| dataset | accuracy | ECE | asks (threshold .05) | VOI flags errors (AUROC) |
|---|---|---|---|---|
| SGD, first user turn: 3,812 conversations, 20 services (2,766 from services unseen in any training) | .942 | .044 | 6% | .84 |
| ABCD, first customer message: 918 conversations, 10 flows | .657 | .186 | 49% | .71 |

- **SGD**: where Lavoir asks, its first guess is right 63% of the time; where it does not ask, 96%.
- **ABCD**: the real agent's first question and the customer's answer lift accuracy by 9.1 points in the
  conversations Lavoir chose to ask about, and by 0.4 points in the others. The value of information is
  concentrated where Lavoir said it would be.

### Single-turn benchmarks

Laya's application suite (400 cases per task, seed 13, same prompts and options), plus MASSIVE-en intent (20
options) and typed-decisions (2,000 decisions). The Laya column is Laya's English checkpoint, which has the same
ModernBERT-large backbone; "Laya best" is the best of its three published checkpoints on each task.

![Single-turn benchmarks](assets/benchmarks.png)

*Accuracy on 12 single-turn classification tasks with no questions asked: Lavoir (orange) against Laya's English
checkpoint (grey), which has the same ModernBERT-large backbone.*

| task | Lavoir acc | Lavoir ECE | Laya (English) acc | Laya (English) ECE | Laya best acc | source in Lavoir's training | source in Laya's training |
|---|---|---|---|---|---|---|---|
| Email spam | **.993** | .014 | .993 | .013 | .993 | train split | yes |
| Phishing | .978 | .031 | **.980** | .012 | .993 | yes, benchmark texts removed | yes |
| AG News | .905 | .045 | **.950** | .032 | .953 | train split | yes |
| Jailbreak (toxic-chat) | **.825** | .035 | .708 | .259 | .763 | no | no |
| Model routing (domain) | **.754** | .086 | .639 | .089 | .659 | related data (train splits) | no |
| RAG relevance (MS MARCO) | **.665** | .046 | .625 | .118 | .658 | train split | yes |
| Toxicity (toxic-chat) | **.605** | .275 | .530 | .296 | .530 | no | no |
| Emotion (DAIR) | .575 | .157 | **.595** | .306 | .600 | no | no |
| Banking77 (77 labels) | **.533** | .130 | .425 | .540 | .493 | no | no |
| Support triage | .383 | .138 | **.503** | .091 | .523 | yes, benchmark texts removed | yes |
| MASSIVE-en intent | **.805** | .139 | .783 | | .783 | no | |
| typed-decisions | **.774** | .157 | .362 | .175 | .766 | train split | train split (typed-decisions checkpoint) |

- Lavoir has the lower ECE on 7 of the 11 tasks where both report one, most clearly on jailbreak (.035 vs .259)
  and Banking77 (.130 vs .540); Laya's is lower on spam, phishing, AG News and support triage.
- typed-decisions: soft accuracy .506, score MAE .223, ECE .157, against Laya's task-specific checkpoint at
  .766 / .471 / .242 / .213.
- Option order: shuffling the options changes the decision in 1% of AG News cases, 5% of emotion, 12% of
  MASSIVE-en and 29% of Banking77 (77 options).
- Lavoir trails Laya on support triage (-12 points) and AG News (-4.5 points), although both models saw these
  sources in training, and slightly on emotion (-2 points).

### Speed

31 ms median, 142 ms p95 for one question on an NVIDIA GH200 (bf16). Laya reports 33–40 ms on a T4.

## Training

1. **Decision phase** from ModernBERT-large: 500 head warm-up steps with a frozen encoder, then 4 epochs; global
   batch 64, encoder LR 2.5e-5, head LR 1e-4, cosine schedule, bf16; soft cross-entropy. Laya's GRPO term
   (`w_rl`) was switched off: it dominated the shared gradients once the decision had converged and slowed down
   the VOI head without improving accuracy.
2. **VOI targets** from that snapshot for every candidate question of every workflow example.
3. **Joint phase**, 2 epochs: decision loss + MSE on the normalized VOI targets, encoder LR 5e-6, Gini-bounded VOI.
4. VOI targets refreshed from the joint snapshot, 1 more joint epoch.
5. Temperatures fitted on a 5% held-out slice: one set for the question-asking workflows, one for general data,
   one per general source (`temperature_by_source` in `config.json`).

98,501 training examples (+5,184 held out); 16 NVIDIA GH200 GPUs, about 80 minutes.

### Data

| part | examples | sources |
|---|---|---|
| Question-asking workflows | 21,601 | 8 synthetic customer-service workflows. Routing rules and answer distributions are exact, so every example's target is the true posterior. Customer messages and answers were written by Qwen3.6-35B-A3B and screened by Gemma-4-26B-A4B for leaked information; each conversation appears at several stages (first message, after one answer, after two). Answers include vague and "I don't know" replies. |
| General single-turn | 19,000 | typed-decisions (train), CLINC150, MultiNLI, Yelp reviews, Schema-Guided Dialogue (train) |
| Laya's application sources | 19,437 | AG News (train), BoolQ, Enron spam (train), phishing emails, MS MARCO v1.1 (train), customer-support tickets; question-form yes/no items built from CLINC150, Yelp, MultiNLI and SGD. Texts used by Laya's benchmark suite were removed. |
| Open task mix | 38,200 | DBpedia, Yahoo Answers, 20 Newsgroups, Tweet Topic, GoEmotions, TweetEval, MATH, Dolly, domain routing (GSM8K, MATH, MBPP, WritingPrompts, Natural Questions, SQL, Dolly), SST-2, IMDB, Civil Comments, hate speech, QQP, PAWS, MRPC, SNLI, ANLI, SciTail, CoLA, Amazon reviews (train splits only) |
| Tool choice | 5,447 | Nemotron post-training tool-calling data: choose the tool the assistant actually called |

Mix by question type: 59% choice, 35% yes/no, 8% score. No benchmark test set (DAIR emotion, toxic-chat,
Banking77, AG News test, typed-decisions test, MASSIVE, ABCD, SGD test) was used for training.

## Intended use

- Routing and triage in support chats, ticket queues and email, where a wrong route is costly and a short
  clarifying question is cheap.
- Deciding *whether* an assistant should ask before acting, and *which* of a fixed set of questions to ask.
- Fast calibrated single-turn classification: moderation, spam, relevance, intent.

Out of scope: generating questions or answers, open-ended dialogue, languages other than English, high-stakes
decisions without human review.

## Limitations

- English only.
- Chooses among the questions you list; it does not write new ones.
- Weaker on workflows unlike the training ones. Plan to fine-tune on a few hundred examples of your workflow; the
  code repository has scripts for decision-only and question-asking fine-tuning.
- Can be overconfident out of distribution (ABCD: ECE .19). VOI is bounded by the model's own uncertainty, so a
  confidently wrong decision is not questioned. Check calibration on your data.
- The question-asking workflows are synthetic. Real users phrase things differently and may answer off-topic.

## License and attribution

The weights are released under CC BY-NC 4.0 because part of the training data is licensed for non-commercial use
only (for example ANLI, MS MARCO and the Yelp reviews). The code is Apache-2.0.

Lavoir follows [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) in its decision-head architecture, input
format and training loss, but uses none of Laya's weights: the decision head and the VOI head were trained from
scratch. The encoder was initialized from [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large)
(Apache-2.0) and trained together with the heads.

## Citation

```bibtex
@misc{yilmaz2026lavoir,
  title        = {Lavoir: A Single-Pass Decision Model That Knows Which Question to Ask},
  author       = {Furkan Yilmaz and Habibe Aleyna Tasdemir and Muhammed Faruk Gozay},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/moganai/lavoir}}
}
```

---

<p align="center">
  <b>Support MoganAI</b><br/>
  If our open models and datasets are useful to you, you can support our work.<br/>
  Açık modellerimiz ve veri setlerimiz işinize yarıyorsa çalışmalarımıza destek olabilirsiniz.
</p>

<p align="center">
  <a href="https://buymeacoffee.com/moganai"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" width="300"/></a>
</p>
