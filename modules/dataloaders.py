import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import MultiImageDataset


class XGDataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        self.dataset = MultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)


    @staticmethod
    def collate_fn(data):
        images_id, image_tag, images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert, seq_lengths, seq_lengths_bert = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)
        max_seq_length_bert = max(seq_lengths_bert)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks


        ids_bert = np.zeros((len(reports_ids_bert), max_seq_length_bert), dtype=int)
        masks_bert = np.zeros((len(reports_masks_bert), max_seq_length_bert), dtype=int)

        for i, report_ids in enumerate(reports_ids_bert):
            ids_bert[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks_bert):
            masks_bert[i, :len(report_masks)] = report_masks

        return images_id, image_tag, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), torch.LongTensor(ids_bert), torch.FloatTensor(masks_bert)
