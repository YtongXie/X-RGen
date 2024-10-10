import logging
import os
from abc import abstractmethod
import cv2
import torch
from modules.utils import generate_heatmap
import json
from tqdm import tqdm

class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader
        self.save_dir = args.save_dir
        self.split = args.split

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with open(os.path.join(self.save_dir, self.split + '.json'), "w") as outfile:
            with torch.no_grad():
                test_gts, test_res = [], []
                test_tags = []
                write_json = {}
                write_json["results"] = []
                write_json["scores"] = []
                for batch_idx, (images_id, image_tags, images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert) in enumerate(tqdm(self.test_dataloader)):

                    images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                        self.device), reports_masks.to(self.device)
                    images, reports_ids, reports_masks, reports_ids_bert, reports_masks_bert = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                        self.device), reports_ids_bert.to(self.device), reports_masks_bert.to(self.device)

                    output, _ = self.model(images, image_tags, mode='sample')
                    reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                    # Data to be written
                    for img_id in range(len(images_id)):
                        dictionary = {
                        "images_id": images_id[img_id],
                        "ground_truths": ground_truths[img_id],
                        "reports": reports[img_id],
                        "image_tag": image_tags[img_id],
                        }

                        # Serializing json
                        write_json["results"].append(dictionary)

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    test_tags.append(image_tags)

                test_tags = ','.join([','.join(w) for w in test_tags]).split(',')
                for topic_t in self.args.topic_type:
                    index_ = [i for i, c in enumerate(test_tags) if c == topic_t]

                    test_met = self.metric_ftns(
                        {i: [gt] for i, gt in enumerate([v for i, v in enumerate(test_gts) if i in index_])},
                        {i: [re] for i, re in enumerate([v for i, v in enumerate(test_res) if i in index_])})
                    write_json["scores"].append({'test_' + topic_t + '_' + k: v for k, v in test_met.items()})
                    log.update(**{'test_' + topic_t + '_' + k: v for k, v in test_met.items()})

                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                log.update(**{'test_' + k: v for k, v in test_met.items()})
                write_json["scores"].append({'test_' + k: v for k, v in test_met.items()})
                json_object = json.dumps(write_json, indent=4)
                outfile.write(json_object)
                print(log)
            return log

    def plot(self):
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0) for layer in
                                     self.model.encoder_decoder.model.decoder.layers]
                for layer_idx, attns in enumerate(attention_weights):
                    assert len(attns) == len(report)
                    for word_idx, (attn, word) in enumerate(zip(attns, report)):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn)
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                                    heatmap)