
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import re
import numpy as np



CE_LABEL_KEYWORDS = {
    'Cardiomegaly':               ['cardiomegaly', 'enlarged heart', 'heart is enlarged', 'cardiac enlargement'],
    'Enlarged Cardiomediastinum': ['widened mediastinum', 'mediastinal widening', 'enlarged mediastinum'],
    'Lung Opacity':               ['opacity', 'opacities', 'opacification'],
    'Lung Lesion':                ['lesion', 'nodule', 'mass'],
    'Edema':                      ['edema', 'vascular congestion', 'pulmonary congestion'],
    'Consolidation':              ['consolidation', 'airspace disease'],
    'Pneumonia':                  ['pneumonia'],
    'Atelectasis':                ['atelectasis', 'atelectatic', 'volume loss'],
    'Pneumothorax':               ['pneumothorax'],
    'Pleural Effusion':           ['effusion'],
    'Pleural Other':              ['pleural thickening', 'pleural scarring'],
    'Fracture':                   ['fracture'],
    'Support Devices':            ['catheter', 'pacemaker', 'picc line', 'hardware', 'sternotomy wire'],
}
CE_LABEL_NAMES = list(CE_LABEL_KEYWORDS.keys())

_NEGATION_CUES = [
    'no ', 'not ', 'without ', 'no evidence of', 'free of', 'negative for',
    'rule out', 'ruled out', 'resolved', 'clear of', 'absence of',
]
_NEGATION_WINDOW_WORDS = 5


def _is_negated(text: str, match_start: int) -> bool:
    """Negation check within the SAME sentence only (bounded by the nearest
    preceding '.'), so a negation cue in an earlier sentence can't
    incorrectly flag an unrelated finding in the next sentence."""
    sentence_start = text.rfind('.', 0, match_start) + 1  # 0 if no prior '.'
    window = text[sentence_start:match_start]
    return any(cue in window for cue in _NEGATION_CUES)


def extract_ce_labels(text: str, label_names=None) -> np.ndarray:
    """Text -> binary vector over CE_LABEL_NAMES (1 = positively mentioned)."""
    label_names = label_names or CE_LABEL_NAMES
    text = str(text).lower()
    vec = np.zeros(len(label_names), dtype=int)
    for i, label in enumerate(label_names):
        for kw in CE_LABEL_KEYWORDS[label]:
            for m in re.finditer(re.escape(kw), text):
                if not _is_negated(text, m.start()):
                    vec[i] = 1
                    break
            if vec[i] == 1:
                break
    return vec


def compute_ce_metrics(gts_texts, res_texts, label_names=None):
    """Precision/Recall/F1 (macro + micro) over CE_LABEL_NAMES, extracted directly
    from report text via extract_ce_labels. Run this BEFORE a full training cycle
    to get an early, cheap signal -- not a substitute for CheXbert-F1/RadGraph-F1."""
    label_names = label_names or CE_LABEL_NAMES
    gt_mat = np.stack([extract_ce_labels(t, label_names) for t in gts_texts])
    res_mat = np.stack([extract_ce_labels(t, label_names) for t in res_texts])

    result = {
        'CE_PRECISION_MICRO': precision_score(gt_mat, res_mat, average='micro', zero_division=0),
        'CE_RECALL_MICRO':    recall_score(gt_mat, res_mat, average='micro', zero_division=0),
        'CE_F1_MICRO':        f1_score(gt_mat, res_mat, average='micro', zero_division=0),
        'CE_PRECISION_MACRO': precision_score(gt_mat, res_mat, average='macro', zero_division=0),
        'CE_RECALL_MACRO':    recall_score(gt_mat, res_mat, average='macro', zero_division=0),
        'CE_F1_MACRO':        f1_score(gt_mat, res_mat, average='macro', zero_division=0),
    }
    return result
# ===================== END CE LABELER =====================




def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

label_set=14
def compute_mlc(gt, pred, label_set):
    res_mlc = {}
    avg_aucroc = 0
    for i, label in enumerate(label_set):
        res_mlc['AUCROC_'] = roc_auc_score(gt[:, i], pred[:, i])
        avg_aucroc += res_mlc['AUCROC_' + label]
    res_mlc['AVG_AUCROC'] = avg_aucroc / len(label_set)

    res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro")
    res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro")
    res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro")
    res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro")
    res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro")
    res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro")

    return res_mlc
# def compute_mlc(gt, pred, label_set):
#     res_mlc = {}
#     avg_aucroc = 0
#     for i, label in enumerate(label_set):
#         res_mlc['AUCROC_' + label] = roc_auc_score(gt[:, i], pred[:, i])
#         avg_aucroc += res_mlc['AUCROC_' + label]
#     res_mlc['AVG_AUCROC'] = avg_aucroc / len(label_set)
#
#     res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro")
#     res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro")
#     res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro")
#     res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro")
#     res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro")
#     res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro")
#
#     return res_mlc


class MetricWrapper(object):
    def __init__(self, label_set):
        self.label_set = label_set

    def __call__(self, gts, res, gts_mlc, res_mlc):
        eval_res = compute_scores(gts, res)
        eval_res_mlc = compute_mlc(gts_mlc, res_mlc, self.label_set)

        eval_res.update(**eval_res_mlc)
        return eval_res
