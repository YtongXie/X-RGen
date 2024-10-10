import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import XGDataLoader
from modules.metrics import compute_scores
from modules.tester import Tester
from modules.loss import compute_loss
from modules.xgren import XGRenModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/images/', help='the path to the directory containing the image.')
    parser.add_argument('--ann_path', type=str, default='data/annotation.json', help='the path to the directory containing the report.')

    # Data loader settings
    parser.add_argument('--use_topic', type=bool, default=True, help='whether use topic.')
    parser.add_argument('--topic_type', type=list, default=['iu', 'knee', 'axr', 'shoulder', 'hip', 'wrist'], choices=['iu', 'knee', 'axr', 'shoulder', 'hip', 'wrist'], help='body parts to be used.')

    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports for decoding.')
    parser.add_argument('--max_seq_length_bert', type=int, default=80, help='the maximum sequence length of the reports for contrastive learning.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used. resnet101 or vit')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for medclip pretrained)
    parser.add_argument('--clip_pretrained_path', type=str, default='models/medclip-pretrained/pytorch_model.bin', help='whether to load the pretrained visual extractor')
    parser.add_argument('--fix_text_encoder', type=bool, default=True, help='if True, fix text encoder. Otherwise, fine-tune text encoder')
    # Model settings (for contras loss)
    parser.add_argument('--contras_loss_w', type=float, default=1.0, help='the weights of contrastive loss., >0 means using kd loss; <=0 means no kd loss')    
    # use ema or not
    parser.add_argument('--use_ema', type=bool, default=False, help='if True, use ema')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features. default: 2048. set 768 if use ViT')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/XRGen_', help='the patch to save the models.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--load', type=str, default='results/XRGen_/model_best.pth', help='whether to load a pre-trained model.')
    parser.add_argument('--split', type=str, default='test', help='whether to load a pre-trained model.')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    test_dataloader = XGDataLoader(args, tokenizer, split=args.split, shuffle=False)

    # build model architecture
    model = XGRenModel(args, tokenizer)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build trainer and start to train
    tester = Tester(model, criterion, metrics, args, test_dataloader)
    # tester.test()
    tester.test()


if __name__ == '__main__':
    main()