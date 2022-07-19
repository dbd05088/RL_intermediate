import argparse


parser = argparse.ArgumentParser("sample")
parser.add_argument('--data', type=str, default='./dataset/cifar10')
parser.add_argument('--log', type=str, default='test')
parser.add_argument('--num_gpus', type=int, default=6)
parser.add_argument('--workers_per_gpu', type=int, default=8)
parser.add_argument('--tr_epoch', type=int, default=256)
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--sample_lr', type=float, default=0.1)

args = parser.parse_args()


