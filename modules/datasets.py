import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.max_seq_length_bert = args.max_seq_length_bert
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.tokenizer_bert = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids_bert'] = self.tokenizer_bert(self.examples[i]['report']).data['input_ids'][:self.max_seq_length_bert]
            self.examples[i]['mask_bert'] = [1] * len(self.examples[i]['ids_bert'])
            
            self.examples[i]['ids'] = self.tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class MultiImageDataset(BaseDataset):
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

        report_ids_bert = example['ids_bert']
        report_masks_bert = example['mask_bert']

        seq_length = len(report_ids)
        seq_length_bert = len(report_ids_bert)
        if 'tag' in example.keys():
            image_tag = example['tag']
            sample = (image_id, image_tag, image, report_ids, report_masks, report_ids_bert, report_masks_bert, seq_length, seq_length_bert)
        else:
            sample = (image_id, image, report_ids, report_masks, report_ids_bert, report_masks_bert, seq_length, seq_length_bert)
        return sample