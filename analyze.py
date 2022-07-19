import torch
import logging.config
import json
import os
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from model import Analyzer



parser = argparse.ArgumentParser("analyzer")
#parser.add_argument('--data', type=str, default='../cifar10_cifar100/cifar10_jh')
parser.add_argument('--data', type=str, default='./dataset/cifar10')
parser.add_argument('--log', type=str, default='test')
parser.add_argument('--tr_epoch', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--model', type=str, default='model.pth')
parser.add_argument('--measure', type=str, default='uncertainty')
parser.add_argument('--good_samples', type=str, default='optimal.json')

args = parser.parse_args()

def main():
    analyzer = Analyzer(train_dir=args.data+'/train')
    if os.path.isfile(args.model):
        print('loding model...')
        analyzer.load_model(args.model)
    else:
        print('training model...')
        analyzer.train_model(args.lr, args.tr_epoch)
        analyzer.save_model(args.model)
    with open(args.good_samples, 'r') as fp:
        optimal = json.load(fp)
    good_samples = optimal['optimal_action']
    print('analyzing...')
    analyzer.analyze(good_samples, args.measure)

if __name__ == '__main__':
    main()
