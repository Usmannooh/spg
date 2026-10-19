
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import torch


class Tokenizer(object):
    def __init__(self, args, device=None):
        self.device        = device
        self.ann_path      = args.ann_path
        self.threshold     = args.threshold
        self.dataset_name  = args.dataset_name
        self.pad_token_id  = 0

        # ── Choose cleaning function ──────────────────────────────────────
        if self.dataset_name == 'iu_xray':
            self.clean_report = self._clean_report_iu_xray
        else:
            self.clean_report = self._clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self._create_vocabulary()

        self._emb_matrix, self.tk_embed_dim = self._build_embedding_matrix(
            args.embedding_path
        )

    def _create_vocabulary(self):
        total_tokens = []
        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab   = sorted(k for k, v in counter.items() if v >= self.threshold)
        vocab  += ['<unk>']

        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token]     = idx + 1   # 0 is reserved for pad
            idx2token[idx + 1]   = token

        return token2idx, idx2token

    def _build_embedding_matrix(self, emb_path: str):

        raw: dict[str, np.ndarray] = {}
        dim: int | None = None

        with open(emb_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                if len(parts) < 2:
                    continue
                token = parts[0]
                vec   = np.array(parts[1:], dtype=np.float32)
                if dim is None:
                    dim = len(vec)
                if len(vec) == dim:       # skip any malformed lines
                    raw[token] = vec

        if dim is None:
            raise ValueError(f"Embedding file appears empty: {emb_path}")
        oov_vec = np.mean(list(raw.values()), axis=0) if raw else np.zeros(dim, dtype=np.float32)
        V   = max(self.idx2token.keys())   # largest token index
        mat = np.zeros((V + 1, dim), dtype=np.float32)   # row 0 = pad
        for idx, token in self.idx2token.items():
            mat[idx] = raw.get(token, oov_vec)
        del raw
        return mat, dim

    def build_embedding_matrix(self) -> np.ndarray:

        return self._emb_matrix   # (V+1, E) numpy float32


    def _clean_report_iu_xray(self, report: str) -> str:
        cleaner = (
            lambda t: t
            .replace('..', '.').replace('..', '.').replace('..', '.')
            .replace('1. ', '')
            .replace('. 2. ', '. ').replace('. 3. ', '. ')
            .replace('. 4. ', '. ').replace('. 5. ', '. ')
            .replace(' 2. ', '. ').replace(' 3. ', '. ')
            .replace(' 4. ', '. ').replace(' 5. ', '. ')
            .strip().lower().split('. ')
        )
        sent_clean = lambda t: re.sub(
            r'[.,?;*!%^&_+():\-\[\]{}]', '',
            t.replace('"', '').replace('/', '').replace('\\', '')
             .replace("'", '').strip().lower()
        )
        sents  = [sent_clean(s) for s in cleaner(report) if sent_clean(s)]
        return ' . '.join(sents) + ' .'

    def _clean_report_mimic_cxr(self, report: str) -> str:
        def _clean(t):
            for old, new in [('\n', ' '), ('__', '_')] * 4 + [('  ', ' ')] * 6 + \
                            [('..', '.')] * 8 + \
                            [('1. ', ''), ('. 2. ', '. '), ('. 3. ', '. '),
                             ('. 4. ', '. '), ('. 5. ', '. '),
                             (' 2. ', '. '), (' 3. ', '. '),
                             (' 4. ', '. '), (' 5. ', '. ')]:
                t = t.replace(old, new)
            return t.strip().lower().split('. ')

        sent_clean = lambda t: re.sub(
            r'[.,?;*!%^&_+():\-\[\]{}]', '',
            t.replace('"', '').replace('/', '').replace('\\', '')
             .replace("'", '').strip().lower()
        )
        sents = [sent_clean(s) for s in _clean(report) if sent_clean(s)]
        return ' . '.join(sents) + ' .'


    def get_vocab_size(self) -> int:
        return len(self.token2idx)

    def get_token_by_id(self, idx: int) -> str:
        return self.idx2token[idx]

    def get_id_by_token(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx.get('<unk>', 0))

    def __call__(self, report: str) -> list[int]:
        """Tokenise a report string → list of integer IDs (BOS/EOS = 0)."""
        tokens = self.clean_report(report).split()
        ids    = [self.get_id_by_token(t) for t in tokens]
        return [0] + ids + [0]   # prepend/append pad (used as BOS/EOS)

    def decode(self, ids) -> str:
        words = []
        for idx in ids:
            if torch.is_tensor(idx):
                idx = idx.item()
            if idx > 0:
                words.append(self.idx2token[idx])
            else:
                break
        return ' '.join(words)

    def decode_batch(self, ids_batch) -> list[str]:
        """Decode a batch of ID sequences."""
        return [self.decode(ids) for ids in ids_batch]
    


