# spg
<!--
 

# SPG: Structured Planning-and-Generation for Radiology Report Generation

Official implementation of **"Beyond LLM Generation: Clinical-Aware Report Selection with Structured Vision-Language Learning for Radiology Report Generation"**.

SPG investigates whether separating report generation from LLM-based candidate evaluation provides a more controlled use of pretrained language models for chest X-ray report generation.

---

## Overview

SPG separates two concerns that are often conflated in LLM-based report generation: **what the model represents** and **how the model decides what to output**.

```
Chest X-ray
  → ResNet-101 visual features
  → Clinical Slot Attention Planner + CRML   (structured clinical representation)
  → Relational-memory Transformer Decoder    (generates C candidate reports)
  → LLM Clinical Quality Selector (CQS)      (scores and selects one candidate)
  → Final report
```

| Component | Role |
|---|---|
| **Clinical Slot Attention Planner** | Decomposes visual evidence into a fixed set of slots to represent co-occurring findings within a study |
| **CRML** (Cross-Batch Rare-Pattern Memory Learning) | Momentum-updated bank of cluster centroids that links similar studies across mini-batches during contrastive training |
| **Transformer Decoder** | R2GenCMN-style relational-memory decoder; generates one beam-search candidate and C-1 temperature-sampled candidates |
| **LLM-CQS** | LoRA-adapted LLM that scores candidates (language confidence + learned quality score) and selects the highest-scoring one. It never generates or rewrites text |
| **Direct LLM Decoder** | Ablation only. LoRA-adapted LLM that generates reports autoregressively from plan tokens |

`clinical_selection_phase` (CQS) and `llm_phase` (Direct LLM Decoder) are mutually exclusive.

---
<img width="1141" height="790" alt="image" src="https://github.com/user-attachments/assets/ea99ce13-0b0a-4e23-a31a-3409856233a4" />


## Repository Structure

```
spg/

├── data/
│   ├── iu_xray/
│   └── mimic_cxr/
├── models/                 # SPG model (planner + CRML + Transformer + CQS / Direct LLM)
├── modules/
│   ├── dataloader/
│   ├── modal/
│   ├── cqs/                # LLM Clinical Quality Selector and reward utilities
│   ├── metrics/
│   ├── tokenizer/
│   └── utils/
├── pycocoevalcap/
├── main_train.py
├── main_test.py
├── test_iu_xray/
├── test_mimic_cxr/
├── train_iu_xray/
└── README.md
```

---

## Requirements

- Python 3.9+
- PyTorch (CUDA recommended)
- `transformers`, `peft`, `bitsandbytes` (LoRA and optional 4-bit LLM loading)
- `scipy`, `numpy`, `pandas`, `scikit-learn`
- Java (required by the METEOR scorer in `pycocoevalcap`)

```bash
git clone https://github.com/Usmannooh/spg.git
cd spg
pip install -r requirements.txt
```

---

## Data Preparation

SPG follows the standard R2Gen data protocol and splits.

**IU X-Ray**
- Download images and reports from [Open-i](https://openi.nlm.nih.gov/)
- Place them under `data/iu_xray/`

**MIMIC-CXR**
- Requires credentialed access on [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/)
- Place images and the Findings-section annotations under `data/mimic_cxr/`

Annotation files follow the R2Gen format. Update the paths in `config/` or pass them as command-line arguments.

---

## Training

SPG is trained in two stages. Run `python main_train.py -h` for the full list of arguments.

### Stage 1: Phase A (visual extractor, planner, CRML, Transformer Decoder)

```bash
python main_train.py \
  --dataset_name iu_xray \
  --save_dir <path/to/save_phase_a> \
  --record_dir <path/to/records_phase_a>
```

### Stage 2: LLM-CQS (Phase A weights are frozen)

```bash
python main_train.py \
  --dataset_name iu_xray \
  --clinical_selection_phase True \
  --llm_phase False \
  --select_num_candidates 4 \
  --phase_a_checkpoint <path/to/phase_a_best.pth> \
  --save_dir <path/to/save_cqs> \
  --record_dir <path/to/records_cqs>
```

### Ablation: Direct LLM Decoder

```bash
python main_train.py \
  --dataset_name iu_xray \
  --llm_phase True \
  --clinical_selection_phase False \
  --phase_a_checkpoint <path/to/phase_a_best.pth> \
  --save_dir <path/to/save_llm> \
  --record_dir <path/to/records_llm>
```

### Outputs

| File | Content |
|---|---|
| `spg_llm.csv` | Per-epoch `train_loss`, `val_*` and `test_*` metrics |
| `captions/captions_epoch_*_test.txt` | Ground truth and prediction for every evaluated sample |
| `model_best.pth` | Best checkpoint by validation metric |



---

## Evaluation

```bash
python main_train.py \
  --dataset_name iu_xray \
  --clinical_selection_phase True \
  --select_num_candidates 4 \
  --load <path/to/model_best.pth>
```

- **NLG metrics:** BLEU-1 to BLEU-4, METEOR, ROUGE-L, CIDEr via `pycocoevalcap`
- **Clinical efficacy:** CheXbert-14 precision, recall and F1. Download the official CheXbert checkpoint and set its path in the evaluation script

---

## Implementation Details

| Setting | Value |
|---|---|
| Visual extractor | ResNet-101 (ImageNet pretrained) |
| Slots | 16 |
| Transformer decoder memory | 2048 x 512, top-k = 32 |
| Optimizer | Adam, lr 5e-5 (visual extractor) / 7e-4 (rest), StepLR (step 20, gamma 0.5) |
| Batch size | 8 |
| LLM (CQS and Direct LLM Decoder) | Qwen2.5-1.5B (base) with LoRA |
| CQS LoRA | rank 32, alpha 64, dropout 0.05 |
| Candidates per study | C = 4 (1 beam search + 3 temperature-sampled) |

---

## Results

Test-set results reported in the paper (SPG with LLM-CQS):

| Dataset | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|
| IU X-Ray | 0.518 | 0.357 | 0.259 | 0.195 | 0.224 | 0.419 | 0.397 |
| MIMIC-CXR | 0.441 | 0.295 | 0.213 | 0.160 | 0.171 | 0.344 | 0.140 |

MIMIC-CXR clinical efficacy (CheXbert-14): Precision 0.677, Recall 0.446, F1 0.538.

These results are for the evaluated datasets, metrics, and experimental setting. See the paper for the full comparison, ablations and limitations.



---

## Citation

```bibtex
@article{usman_spg,
  title   = {Beyond LLM Generation: Clinical-Aware Report Selection with Structured Vision-Language Learning for Radiology Report Generation},
  author  = {Usman, Muhammad and Chen, Chengbin and Zhang, Yijia},
  note    = {Manuscript under preparation}
}
```

---

## Acknowledgements

This work was supported by the National Natural Science Foundation of China under Grant 62572089. The Transformer decoder builds on R2Gen and R2GenCMN.





-->
