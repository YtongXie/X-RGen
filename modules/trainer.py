import os
from abc import abstractmethod
import numpy as np
import time
import torch
import pandas as pd
from numpy import inf
from tqdm import tqdm
from torch import nn
import wandb
from timm.utils import ModelEmaV2


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        if self.args.use_ema:
            self.model_ema = ModelEmaV2(model, decay=0.999)
            self.model_ema.module.to(self.device)
            print('################################# using ema')
        else:
            self.model_ema = None

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

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

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
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
        if self.args.use_ema:
            state['state_dict_ema'] = self.model_ema.module.state_dict()
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.args.use_ema:
            self.model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.args = args

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def _eval_epoch(self, log, split, dataloader):

        if self.args.use_ema:
            model = self.model_ema.module
        else:
            model = self.model

        model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            val_contras_loss = 0
            val_cap_loss = 0
            val_loss = 0
            val_tags = []
            for batch_idx, (images_id, image_tags, images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert) in enumerate(tqdm(dataloader)):
                images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device), reports_ids_bert.to(self.device), reports_masks_bert.to(self.device)

                if self.args.contras_loss_w > 0:
                    output, _ = model(images, image_tags, mode='sample')
                    ############## print val loss
                    val_output, val_logits_per_text = model(images, image_tags, reports_ids, reports_ids_bert, reports_masks_bert, mode='train')
                    # contrastive loss
                    contras_loss = self.clip_loss(val_logits_per_text)
                    # image caption loss
                    caption_loss = self.criterion(val_output, reports_ids, reports_masks)
                    loss = contras_loss + caption_loss
                    val_contras_loss += contras_loss.item()
                    val_cap_loss += caption_loss.item()
                    val_loss += loss.item()

                else:
                    output, _ = model(images, image_tags, mode='sample')
                    ############## print val loss
                    val_output, _ = model(images, image_tags, reports_ids, reports_ids_bert, reports_masks_bert, mode='train')
                    # image caption loss
                    caption_loss = self.criterion(val_output, reports_ids, reports_masks)
                    val_cap_loss += caption_loss.item()

                # if batch_idx>2:
                #     break

                reports = model.tokenizer.decode_batch(output.cpu().numpy())
                # print(reports)
                ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                val_res.extend(reports)
                val_gts.extend(ground_truths)
                val_tags.append(image_tags)

            val_tags = ','.join([','.join(w) for w in val_tags]).split(',')
            for topic_t in self.args.topic_type:
                index_ = [i for i, c in enumerate(val_tags) if c == topic_t]

                val_met = self.metric_ftns(
                    {i: [gt] for i, gt in enumerate([v for i, v in enumerate(val_gts) if i in index_])},
                    {i: [re] for i, re in enumerate([v for i, v in enumerate(val_res) if i in index_])})
                wandb.log({split + '_' + topic_t + '_' + k: v for k, v in val_met.items()})
                log.update(**{split + '_' + topic_t + '_' + k: v for k, v in val_met.items()})

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            wandb.log({split + '_' + k: v for k, v in val_met.items()})
            wandb.log({split + "_caption_loss_": val_cap_loss / len(dataloader)})
            wandb.log({split + "_contras_loss_": val_contras_loss / len(dataloader)})
            wandb.log({split + "_loss_": val_loss / len(dataloader)})
            log.update(**{split + '_' + k: v for k, v in val_met.items()})
            log.update(**{split + "_caption_loss_": val_cap_loss / len(dataloader)})
            log.update(**{split + "_contras_loss_": val_contras_loss / len(dataloader)})
            log.update(**{split + "_loss_": val_loss / len(dataloader)})

    def _train_epoch(self, epoch):

        train_contras_loss = 0
        train_cap_loss = 0
        train_loss = 0

        self.model.train()
        for batch_idx, (images_id, image_tags, images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert) in enumerate(
                tqdm(self.train_dataloader)):
            images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device), reports_ids_bert.to(self.device), reports_masks_bert.to(self.device)

            if self.args.contras_loss_w > 0:
                output, logits_per_text = self.model(images, image_tags, reports_ids, reports_ids_bert, reports_masks_bert, mode='train')
                # contrastive loss
                contras_loss = self.clip_loss(logits_per_text) * self.args.contras_loss_w
                # image caption loss
                caption_loss = self.criterion(output, reports_ids, reports_masks)
                loss = contras_loss + caption_loss
                train_contras_loss += contras_loss.item()
                train_cap_loss += caption_loss.item()
                train_loss += loss.item()

            else:
                output, logits_per_text = self.model(images, image_tags, reports_ids, mode='train')
                # image caption loss
                loss = self.criterion(output, reports_ids, reports_masks)
                train_cap_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            if self.args.use_ema:
                self.model_ema.update(self.model)

            # if batch_idx>2:
            #     break

        log = {'train_contras_loss': train_contras_loss / len(self.train_dataloader),
               'train_caption_loss': train_cap_loss / len(self.train_dataloader),
               'train_loss': train_loss / len(self.train_dataloader)}
        wandb.log({"train_contras_loss": train_contras_loss / len(self.train_dataloader)})
        wandb.log({"train_caption_loss": train_cap_loss / len(self.train_dataloader)})
        wandb.log({"train_loss": train_loss / len(self.train_dataloader)})

        # log = {}
        self._eval_epoch(log, 'val', self.val_dataloader)
        self._eval_epoch(log, 'test', self.test_dataloader)

        self.lr_scheduler.step()

        return log
