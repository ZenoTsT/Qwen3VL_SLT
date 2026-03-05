from typing import List, Dict

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer


def compute_bleu_1_4(preds: List[str], refs: List[str]) -> Dict[str, float]:
    out = {}

    for n in [1, 2, 3, 4]:
        bleu = BLEU(max_ngram_order=n, smooth_method="exp") # smooth helps on short sentences
        score = bleu.corpus_score(preds, [refs]).score      # I pass [refs] since bleu support more that one ref for each sentence
        out[f"bleu{n}"] = float(score)

    return out


def compute_rouge_l(preds: List[str], refs: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    total = 0.0
    n = max(1, len(preds))

    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        total += s["rougeL"].fmeasure

    return float(total / n)


def compute_all(preds: List[str], refs: List[str]) -> Dict[str, float]:
    out = {}
    out.update(compute_bleu_1_4(preds, refs))
    out["rougeL_f"] = compute_rouge_l(preds, refs)
    return out