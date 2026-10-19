
import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from modules import utils



class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir      = args.image_dir
        self.ann_path       = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split          = split
        self.tokenizer      = tokenizer
        self.transform      = transform
        self.ann            = json.loads(open(self.ann_path, 'r').read())

        raw_examples     = self.ann[self.split]
        self.examples    = []
        seen_image_paths = set()

        for example in raw_examples:
            image_paths = example['image_path']
            valid = True
            for p in image_paths:
                img_path = os.path.join(self.image_dir, p)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    valid = False
                    break
            if not valid:
                continue

            img_key = tuple(image_paths)
            if img_key in seen_image_paths:
                continue
            seen_image_paths.add(img_key)
            self.examples.append(example)
        if len(self.examples) == 0:
            self.examples = raw_examples
        for i in range(len(self.examples)):
            self.examples[i]['ids']  = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

class BaseMimicDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir      = args.image_dir
        self.ann_path       = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split          = split
        self.tokenizer      = tokenizer
        self.transform      = transform
        self.ann            = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids']  = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        return image_id, image, report_ids, report_masks, seq_length, example['report']

class MimiccxrSingleImageDataset(BaseMimicDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        return image_id, image, report_ids, report_masks, seq_length, example['report']


class BaseDatasetProgressive(Dataset):
    def __init__(self, args, tokenizer, split, transform=None, limit_length=None):
        self.image_dir      = args.image_dir
        self.ann_path       = args.ann_path
        self.vocab_path     = args.vocab_path
        self.transform      = transform
        self.max_seq_length = args.max_seq_length
        self.src_max_seq_length = getattr(args, 'src_max_seq_length', 100)
        self.tgt_max_seq_length = getattr(args, 'tgt_max_seq_length', 100)
        self.split          = split
        self.tokenizer      = tokenizer
        self.clean_report   = utils.clean_report_mimic_cxr
        self.ann            = json.loads(open(self.ann_path, 'r').read())

        if isinstance(split, list):
            self.examples = self.get_folds()
        else:
            self.examples = self.ann[self.split]

        if getattr(args, 'normal_abnormal', None) is not None:
            self.examples = [ex for ex in self.examples if ex['abnormal'] == args.normal_abnormal]
        if limit_length is not None:
            self.examples = self.examples[:limit_length]

        self.report_mode = args.report_mode
        for i in range(len(self.examples)):
            self.examples[i]['ids']  = tokenizer(self.examples[i][self.report_mode])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            input_txt      = f"{self.clean_report(self.examples[i][self.report_mode])}"
            decoder_input  = f"</s><s>{self.clean_report(self.examples[i]['report'])}"
            label          = f"<s>{self.clean_report(self.examples[i]['report'])}</s>"
            self.examples[i]['input_bart']    = input_txt
            self.examples[i]['decoder_input'] = decoder_input
            self.examples[i]['label']         = label

    def __len__(self):
        return len(self.examples)

    def get_folds(self):
        examples = []
        for x in self.ann:
            if str(x['fold']) in self.split:
                examples.append(x)
        return examples


class IuxrayDatasetProgressive(BaseDatasetProgressive):
    def __getitem__(self, idx):
        example    = self.examples[idx]
        image_id   = example['id']
        image_path = example['image_path']

        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        image        = torch.stack((image_1, image_2), 0)
        report_ids   = example['ids']
        report_masks = example['mask']
        seq_length   = len(report_ids)
        input_bart         = example['input_bart']
        decoder_input_bart = example['decoder_input']
        label_bart         = example['label']

        return (int(float(image_id)), image, report_ids, report_masks, seq_length,
                input_bart, decoder_input_bart, label_bart)



