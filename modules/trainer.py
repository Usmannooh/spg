import logging
import os
import json
from abc import abstractmethod
import random
import re
import signal
import time
import torch
from numpy import inf
import pandas as pd
from torch.utils.data import Subset, DataLoader


class TimeoutException(Exception):
    pass
def _timeout_handler(signum, frame):
    raise TimeoutException("METEOR timeout")
def _compute_metrics_with_timeout(gts_dict, res_dict, metric_ftns, timeout_sec=300):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    metrics = {}
    # BLEU (verbose=0 works)
    bleu_scorer = Bleu(4)
    score, _ = bleu_scorer.compute_score(gts_dict, res_dict, verbose=0)
    metrics["BLEU_1"] = score[0]
    metrics["BLEU_2"] = score[1]
    metrics["BLEU_3"] = score[2]
    metrics["BLEU_4"] = score[3]
    ## METEOR with timeout (slow — Java process)
    try:
        meteor_scorer = Meteor()
        score, _ = meteor_scorer.compute_score(gts_dict, res_dict)
        metrics["METEOR"] = score
        meteor_scorer.__del__()
    except TimeoutException:
        logging.warning(f" METEOR timeout after {timeout_sec}s — skipped this epoch")
        metrics["METEOR"] = 0.0
    except Exception as e:
        logging.warning(f" METEOR error: {e} — skipped")
        metrics["METEOR"] = 0.0
    rouge_scorer = Rouge()
    score, _ = rouge_scorer.compute_score(gts_dict, res_dict)
    metrics["ROUGE_L"] = score
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts_dict, res_dict)
    metrics["CIDEr"] = score
    return metrics
def _normalize_report(text: str) -> str:
    """Standardize text for fair n-gram metric computation."""
    text = str(text).strip().lower()
    # "normal ."  →  "normal."
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # multiple spaces → single space
    text = re.sub(r'\s+', ' ', text)
    return text


def _load_synonym_map(path):

    if not path or not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        raw_map = json.load(f)
    pairs = []
    for canonical, synonyms in raw_map.items():
        if canonical.startswith('_') or not isinstance(synonyms, list):
            continue
        for syn in synonyms:
            pairs.append((syn.lower().strip(), canonical.lower().strip()))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    compiled = [(re.compile(r'\b' + re.escape(syn) + r'\b'), canon) for syn, canon in pairs]
    return compiled


def _canonicalize_synonyms(text: str, compiled_pairs) -> str:

    for pattern, canonical in compiled_pairs:
        text = pattern.sub(canonical, text)
    return text

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler('training_log_spg_llm.txt')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.record_dir = args.record_dir
        self.caption_dir = os.path.join(self.record_dir, 'captions')
        os.makedirs(self.caption_dir, exist_ok=True)
        self.best_recorder = {
            'val': {self.mnt_metric: self.mnt_best},
            'test': {self.mnt_metric_test: self.mnt_best}
        }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if args.resume is not None:
            self._resume_checkpoint(args.resume)
        self.eval_subset_size = getattr(args, 'eval_subset_size', 5000)
        self.full_eval_period = getattr(args, 'full_eval_period', 5)

        # Metric-time-only synonym canonicalization (see _canonicalize_synonyms
        # docstring). Empty/absent JSON => no-op, metrics behave as before.
        self.synonym_pairs = _load_synonym_map(args.report_synonym_path)
        if self.synonym_pairs:
            self.logger.info(f"[EVAL] Loaded {len(self.synonym_pairs)} synonym mappings from "
                              f"{args.report_synonym_path}")

        self.cmn_curriculum_epochs = args.cmn_curriculum_epochs
        self.cmn_train_captions = None
        self.select_candidates_cache = None
        self.select_eval_cache = {}


        self.use_grad_accum = bool(args.use_grad_accum)
        self.grad_accum_steps = max(1, int(args.grad_accum_steps))
        if not self.use_grad_accum:
            self.grad_accum_steps = 1
        self.logger.info(
            f"[TRAIN CONFIG] Gradient Accumulation: "
            f"{'ON, Steps=' + str(self.grad_accum_steps) if self.use_grad_accum else 'OFF'}"
        )


    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _save_captions_to_txt(self, epoch, split, captions_data):
        txt_filename = f"captions_epoch_{epoch}_{split}.txt"
        txt_path = os.path.join(self.caption_dir, txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Epoch {epoch} - {split.upper()} Set Caption Results ===\n")
            f.write(f"Generated at: {time.ctime()}\n")
            f.write(f"Total samples: {len(captions_data)}\n\n")
            for idx, data in enumerate(captions_data, 1):
                f.write(f"Sample {idx}\n")
                f.write(f"Image ID: {data['image_id']}\n")
                f.write(f"Ground Truth: {data['ground_truth']}\n")
                if 'cmn_draft' in data:
                    f.write(f"Transformer Draft: {data['cmn_draft']}\n")
                f.write(f"Prediction: {data['prediction']}\n")
                f.write("-" * 80 + "\n\n")
        self.logger.info(f"Saved {split} set captions to: {txt_path}")

    def _get_select_candidates_for_batch(self, images_id, images, cache):

        ids_str = [str(i) for i in images_id]
        missing = [i for i, iid in enumerate(ids_str) if iid not in cache]
        if missing:
            with torch.no_grad():
                sel = images[missing]
                if self.args.dataset_name == 'iu_xray':
                    att_feats_0, fc_feats_0, _ = self.model.visual_extractor(sel[:, 0])
                    att_feats_1, fc_feats_1, _ = self.model.visual_extractor(sel[:, 1])
                    att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
                    fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
                else:
                    att_feats, fc_feats, _ = self.model.visual_extractor(sel)
                plan_tokens = self.model.planner(att_feats)
                att_feats_ext = torch.cat([att_feats, plan_tokens], dim=1) if self.model.use_planner else att_feats
                new_cands = self.model._generate_cmn_candidates(
                    fc_feats, att_feats_ext, self.model.select_num_candidates
                )
            for local_i, global_i in enumerate(missing):
                cache[ids_str[global_i]] = new_cands[local_i]
        return [cache[iid] for iid in ids_str]

    def _evaluate_split(self, dataloader, split, epoch, subset=5000):
        dataset = dataloader.dataset
        total_size = len(dataset)
        eval_indices = list(range(total_size))
        if subset is not None and total_size > subset:
            rng = random.Random(self.args.seed)
            eval_indices = rng.sample(range(total_size), subset)
            self.logger.info(
                f"[EVAL] {split.upper()}  {subset}/{total_size} samples (seed={self.args.seed})"
            )
        subset_ds = Subset(dataset, eval_indices)
        eval_loader = DataLoader(
            subset_ds,
            batch_size=dataloader.batch_size,
            shuffle=False,
            collate_fn=dataloader.collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        gts, res = [], []
        captions_data = []
        total_batches = len(eval_loader)
        log_every = max(1, total_batches // 5)
        t_start = time.time()

        is_llm_text_phase = self.args.llm_phase
        is_clinical_selection_phase = self.args.clinical_selection_phase

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):

                if len(batch) == 5:
                    images_id, images, reports_ids, reports_masks, raw_reports = batch
                else:
                    images_id, images, reports_ids, reports_masks = batch
                    raw_reports = None

                if batch_idx % log_every == 0:
                    elapsed = time.time() - t_start
                    done = batch_idx * dataloader.batch_size
                    self.logger.info(
                        f"  [{split}] {min(done, len(eval_indices))}/{len(eval_indices)} "
                        f"| batch {batch_idx}/{total_batches} | {elapsed:.0f}s"
                    )

                images = images.to(self.device)
                if is_clinical_selection_phase:
                    batch_candidates = self._get_select_candidates_for_batch(
                        images_id, images, self.select_eval_cache
                    )
                    reports_raw, _ = self.model(images, mode='sample', cached_candidates=batch_candidates)
                else:
                    reports_raw, _ = self.model(images, mode='sample')

                # Ground truth selection
                if is_llm_text_phase and raw_reports is not None:
                    gt_raw = list(raw_reports)
                else:
                    gt_raw = self.model.tokenizer.decode_batch(
                        reports_ids[:, 1:].cpu().numpy()
                    )


                if is_llm_text_phase:
                    reports_norm = [_normalize_report(r) for r in reports_raw]
                    gt_norm = [_normalize_report(g) for g in gt_raw]
                else:
                    reports_norm = reports_raw
                    gt_norm = gt_raw


                if self.synonym_pairs and is_llm_text_phase:
                    reports_norm = [_canonicalize_synonyms(r, self.synonym_pairs) for r in reports_norm]
                    gt_norm = [_canonicalize_synonyms(g, self.synonym_pairs) for g in gt_norm]

                # Captions file ke liye original text
                for i, (img_id, gt, pred) in enumerate(zip(images_id, gt_raw, reports_raw)):
                    entry = {
                        'image_id': img_id,
                        'ground_truth': gt.strip(),
                        'prediction': pred.strip()
                    }
                    captions_data.append(entry)

                # Metrics ke liye normalized text
                for gt, pred in zip(gt_norm, reports_norm):
                    res.append(pred)
                    gts.append(gt)

        elapsed_total = time.time() - t_start
        self.logger.info(
            f"[EVAL] {split.upper()} done — {len(res)} samples in {elapsed_total:.1f}s"
        )
        self._save_captions_to_txt(epoch, split, captions_data)
        gts_dict = {i: [gt] for i, gt in enumerate(gts)}
        res_dict = {i: [re] for i, re in enumerate(res)}
        metrics = _compute_metrics_with_timeout(gts_dict, res_dict, self.metric_ftns)

        return metrics, gts, res
    def train(self):
        not_improved_count = 0
        best_epoch = 0
        df = None
        os.makedirs(self.record_dir, exist_ok=True)
        filename = os.path.join(self.record_dir, 'spg_llm.csv')
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(result)
            if df is None:
                df = pd.DataFrame.from_dict(log, orient='index').T
            else:
                df = pd.concat([df, pd.DataFrame.from_records([log])], ignore_index=True)
            df.to_csv(filename, index=False)
            self._record_best(log)
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (
                            (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or
                            (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    )
                except KeyError:
                    self.logger.warning(f"Metric '{self.mnt_metric}' not found.")
                    self.mnt_mode = 'off'
                    improved = False
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    best_epoch = epoch
                else:
                    not_improved_count += 1
                if not_improved_count > self.early_stop:
                    self.logger.info(f"No improvement for {self.early_stop} epochs. Stopping.")
                    break
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            print(f'Best performance in epoch: {best_epoch}')
        self.logger.info("Running FINAL FULL evaluation on best model...")
        self._final_full_evaluation()
        self._print_best()
        self._print_best_to_file()

    def _final_full_evaluation(self):
        best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info("Best model loaded for final evaluation.")
        self.model.eval()
        self.logger.info("Final FULL val evaluation...")
        val_met, _, _ = self._evaluate_split(self.val_dataloader, 'val', epoch=-1, subset=None)
        self.logger.info("Final FULL test evaluation...")
        test_met, test_gts, test_res = self._evaluate_split(self.test_dataloader, 'test', epoch=-1, subset=None)
        test_res_df = pd.DataFrame(test_res)
        test_gts_df = pd.DataFrame(test_gts)
        test_res_df.to_csv(os.path.join(self.args.save_dir, "final_res.csv"), index=False, header=False)
        test_gts_df.to_csv(os.path.join(self.args.save_dir, "final_gts.csv"), index=False, header=False)
        self.logger.info(f"FINAL VAL: {val_met}")
        self.logger.info(f"FINAL TEST: {test_met}")

    def _record_best(self, log):
        improved_val = (
                (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][self.mnt_metric]) or
                (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][
                    self.mnt_metric])
        )
        if improved_val:
            self.best_recorder['val'].update(log)
        improved_test = (
                (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
                    self.mnt_metric_test]) or
                (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                    self.mnt_metric_test])
        )
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info(f'Best val results (w.r.t {self.args.monitor_metric}):')
        for k, v in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(k), v))
        self.logger.info(f'Best test results (w.r.t {self.args.monitor_metric}):')
        for k, v in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(k), v))

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        for rec in ('val', 'test'):
            self.best_recorder[rec]['time'] = crt_time
            self.best_recorder[rec]['seed'] = self.args.seed
            self.best_recorder[rec]['best_model_from'] = rec
        os.makedirs(self.args.record_dir, exist_ok=True)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
        record_table = pd.read_csv(record_path) if os.path.exists(record_path) else pd.DataFrame()
        record_table = pd.concat(
            [record_table,
             pd.DataFrame([self.best_recorder['val']]),
             pd.DataFrame([self.best_recorder['test']])],
            ignore_index=True
        )
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("No GPU. Training on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(f"Requested {n_gpu_use} GPUs, only {n_gpu} available.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        path = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, path)
        self.logger.info(f"Checkpoint saved: {path}")
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Best model saved: model_best.pth")

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f"Loading checkpoint: {resume_path}")
        checkpoint = torch.load(str(resume_path))
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info(f"Resumed from epoch {self.start_epoch}")

# TRAINER
class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args,
                 lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super().__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def save_result(self, filename, epoch, *context):
        with open(filename, 'a') as f:
            seq = '\t'
            filen = filename.split('/')[-1]
            if filen == 'train.txt':
                if epoch == 1:
                    f.write(f'epoch{seq}loss{seq}\n')
                f.write(seq.join(str(c) for c in context) + '\n')
            else:
                if epoch == 1:
                    f.write('epoch' + seq + seq.join(context[0].keys()) + '\n')
                f.write(str(epoch) + seq + seq.join(str(v) for v in context[0].values()) + '\n')

    def _precompute_cmn_train_captions(self):

        self.logger.info("[CURRICULUM] Precomputing Transformer captions on train split ...")
        was_training = self.model.training
        self.model.eval()
        cache = {}
        with torch.no_grad():
            for batch in self.train_dataloader:
                if len(batch) == 5:
                    images_id, images, reports_ids, reports_masks, _ = batch
                else:
                    images_id, images, reports_ids, reports_masks = batch
                images = images.to(self.device)

                if self.args.dataset_name == 'iu_xray':
                    att_feats_0, fc_feats_0, _ = self.model.visual_extractor(images[:, 0])
                    att_feats_1, fc_feats_1, _ = self.model.visual_extractor(images[:, 1])
                    att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
                    fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
                else:
                    att_feats, fc_feats, _ = self.model.visual_extractor(images)

                plan_tokens = self.model.planner(att_feats)
                att_feats_ext = torch.cat([att_feats, plan_tokens], dim=1) if self.model.use_planner else att_feats

                output_tokens, _ = self.model.encoder_decoder(
                    fc_feats, att_feats_ext, mode='sample', update_opts={}
                )
                captions = [self.model.tokenizer.decode(seq) for seq in output_tokens]
                for img_id, cap in zip(images_id, captions):
                    cache[str(img_id)] = cap

        if was_training:
            self.model.train()
        self.logger.info(f"[CURRICULUM] Cached {len(cache)} Transformer train captions.")
        self.cmn_train_captions = cache

    def _precompute_select_candidates(self):
        self.logger.info("[select] Precomputing Transformer candidate pool on train split...")

        was_training = self.model.training
        self.model.eval()

        train_ds = self.train_dataloader.dataset
        original_transform = train_ds.transform

        # IMPORTANT: deterministic eval transform use karo
        train_ds.transform = self.val_dataloader.dataset.transform

        cache = {}
        t0 = time.time()
        total_batches = len(self.train_dataloader)
        log_every = max(1, total_batches // 10)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_dataloader):
                if len(batch) == 5:
                    images_id, images, reports_ids, reports_masks, _ = batch
                else:
                    images_id, images, reports_ids, reports_masks = batch

                images = images.to(self.device)

                self._get_select_candidates_for_batch(
                    images_id, images, cache
                )

                if batch_idx % log_every == 0:
                    self.logger.info(
                        f"  [select precompute] {batch_idx}/{total_batches} batches "
                        f"| {time.time() - t0:.0f}s elapsed"
                    )

        train_ds.transform = original_transform

        if was_training:
            self.model.train()

        self.logger.info(
            f"[select] Cached candidates for {len(cache)} train images in {time.time() - t0:.0f}s."
        )

        self.select_candidates_cache = cache

    def _train_epoch(self, epoch):
        self.logger.info(f'[{epoch}/{self.epochs}] Training started.')
        train_loss = 0
        self.model.train()

        is_clinical_selection_phase = self.args.clinical_selection_phase
        if self.args.llm_phase and hasattr(self.model, 'llm_decoder'):
            warmup_epochs = self.args.projector_warmup_epochs
            lora_should_be_trainable = epoch > warmup_epochs
            self.model.llm_decoder.set_lora_trainable(lora_should_be_trainable)
            if epoch == 1 or epoch == warmup_epochs + 1:
                state = "UNFROZEN (joint training)" if lora_should_be_trainable else "FROZEN (projector warmup)"
                self.logger.info(f"[{epoch}/{self.epochs}] LLM LoRA is {state}")
        elif is_clinical_selection_phase and getattr(self.model, 'selecter', None) is not None:

            warmup_epochs = self.args.projector_warmup_epochs
            lora_should_be_trainable = epoch > warmup_epochs
            self.model.selector.set_lora_trainable(lora_should_be_trainable)
            if epoch == 1 or epoch == warmup_epochs + 1:
                state = "UNFROZEN (joint training)" if lora_should_be_trainable else "FROZEN (projector warmup)"
                self.logger.info(f"[{epoch}/{self.epochs}] Clinical selector LoRA is {state}")

        is_llm_phase = self.args.llm_phase
        if is_llm_phase and self.cmn_curriculum_epochs > 0 \
                and epoch <= self.cmn_curriculum_epochs and self.cmn_train_captions is None:
            self._precompute_cmn_train_captions()

        if is_clinical_selection_phase and self.select_candidates_cache is None:
            self._precompute_select_candidates()

        total_batches = len(self.train_dataloader)
        log_period = max(1, min(self.args.log_period, total_batches // 4))

        accum_steps = self.grad_accum_steps
        trainable_params = list(self.model.parameters())
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_dataloader):
            # BACKWARD COMPATIBLE
            if len(batch) == 5:
                images_id, images, reports_ids, reports_masks, report_texts = batch
            else:
                images_id, images, reports_ids, reports_masks = batch
                report_texts = None

            images = images.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)

            # Build the LLM's training target ONLY -- reports_ids (Transformer's own
            # supervised target, used above) is never touched by any of this.
            report_texts_for_llm = report_texts
            if (is_llm_phase or is_clinical_selection_phase) and report_texts is not None:
                if self.cmn_curriculum_epochs > 0 and epoch <= self.cmn_curriculum_epochs \
                        and self.cmn_train_captions is not None:
                    source_texts = [self.cmn_train_captions.get(str(iid), rt)
                                     for iid, rt in zip(images_id, report_texts)]
                else:
                    source_texts = report_texts
                if self.synonym_pairs:
                    report_texts_for_llm = [_canonicalize_synonyms(_normalize_report(t), self.synonym_pairs)
                                             for t in source_texts]
                else:
                    report_texts_for_llm = [_normalize_report(t) for t in source_texts]

            # PHASE B HANDLING
            if is_llm_phase:
                # Phase B: model ko raw texts chahiye, aur model directly scalar LLM loss return karta hai
                output, crml_loss = self.model(
                    images, reports_ids, report_texts=report_texts_for_llm, mode='train'
                )
                loss = output
            elif is_clinical_selection_phase:
                batch_candidates = self._get_select_candidates_for_batch(
                    images_id, images, self.select_candidates_cache
                )
                output, crml_loss = self.model(
                    images, reports_ids, report_texts=report_texts_for_llm,
                    mode='train', cached_candidates=batch_candidates
                )
                loss = output
            else:
                # Plain Phase A (Transformer only)
                output, crml_loss = self.model(images, reports_ids, mode='train')
                loss = 0.9 * self.criterion(output, reports_ids, reports_masks) + 0.1 * crml_loss

            (loss / accum_steps).backward()
            train_loss += loss.item()

            is_update_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == total_batches)
            if is_update_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if is_clinical_selection_phase and batch_idx % log_period == 0:
                    # Diagnostic only.
                    with torch.no_grad():
                        plan_tokens = self.model.compute_plan_tokens(images)[2]
                        C = len(batch_candidates[0])
                        scores = self.model.selector._score_batch(
                            plan_tokens.repeat_interleave(C, 0),
                            [c for cands in batch_candidates for c in cands]
                        )[0].view(-1, C)
                        self.logger.info(
                            f"[{epoch}/{self.epochs}] step {batch_idx}: sample candidate scores = "
                            f"{scores[0].cpu().numpy()}"
                        )

            if batch_idx % log_period == 0:
                avg = train_loss / (batch_idx + 1)
                accum_note = f" (accum {(batch_idx % accum_steps) + 1}/{accum_steps})" if accum_steps > 1 else ""

                self.logger.info(
                    f'[{epoch}/{self.epochs}] step {batch_idx}/{total_batches} '
                    f'| loss={avg:.5f}  crml={crml_loss.item():.5f}{accum_note}'

                )

        log = {'train_loss': train_loss / total_batches}



        # Evaluation — same subset size for val and test
        is_full_eval = (epoch % self.full_eval_period == 0) or (epoch == self.epochs)
        subset_size = None if is_full_eval else self.eval_subset_size
        eval_label = "FULL" if is_full_eval else f"SUBSET-{subset_size}"
        self.logger.info(f'[{epoch}/{self.epochs}] VAL evaluation ({eval_label}).')
        self.model.eval()
        val_met, _, _ = self._evaluate_split(self.val_dataloader, 'val', epoch, subset=subset_size)
        log.update(**{f'val_{k}': v for k, v in val_met.items()})
        self.logger.info(f'[{epoch}/{self.epochs}] TEST evaluation ({eval_label}).')
        test_met, test_gts, test_res = self._evaluate_split(self.test_dataloader, 'test', epoch, subset=subset_size)
        log.update(**{f'test_{k}': v for k, v in test_met.items()})
        self.save_result(self.checkpoint_dir + 'test.txt', epoch, test_met)
        test_res_df = pd.DataFrame(test_res)
        test_gts_df = pd.DataFrame(test_gts)
        test_res_df.to_csv(os.path.join(self.args.save_dir, "res.csv"), index=False, header=False)
        test_gts_df.to_csv(os.path.join(self.args.save_dir, "gts.csv"), index=False, header=False)
        self.lr_scheduler.step()
        return log