import argparse
import os
import logging
import logging.config
import json

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from configs import args
from model import BanditModel

def main():
    writer = SummaryWriter(log_dir='exps_tblog/' + args.log + '/{:%m-%d %H:%M:%S}'.format(datetime.now()))
    os.makedirs('exps/' + args.log, exist_ok=True)
    filename = 'exps/' + args.log + '/{:%m-%d %H:%M:%S}.txt'.format(datetime.now())
    #logging.config.fileConfig('../CIL-master/configuration/logging.conf', defaults={'logfilename': filename})
    logger = logging.getLogger('Class Incremental Learning')
    logger.info(args)
    RL_module = BanditModel(train_dir=args.data+'/train', test_dir=args.data+'/test', num_gpus=args.num_gpus,
                            workers_per_gpu=args.workers_per_gpu, logger=logger, writer=writer, lr=args.lr,
                            max_iter=args.max_iter, tr_epoch=args.tr_epoch, sample_lr=args.sample_lr)
    best_act, best_rew, opt_act, opt_rew = RL_module.learn()
    best_act = [act.tolist() for act in best_act]
    opt_act = [act.tolist() for act in opt_act]
    data = {'best_action': best_act, 'best_reward': best_rew, 'optimal_action': opt_act, 'optimal_reward': opt_rew}

    logger.info(data)

    with open(filename[:-4]+'.json', 'w') as fp:
        json.dump(data, fp)


if __name__ == '__main__':
    main()
