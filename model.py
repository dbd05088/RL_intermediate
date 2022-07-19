import os
import torch
import multiprocessing

import numpy as np
import torch.nn as nn
import ray
import torch.nn.functional as F
from easydict import EasyDict as edict
from torchvision import transforms
from torch.utils.data import DataLoader

from resnet import ResNet
from dataloader import SampleDataset, TestDataset
from utils import get_statistics, select_optimizer
from augment import CIFAR10Policy, cutmix_data
from sklearn.manifold import TSNE

from configs import args

ray.init(num_gpus=args.num_gpus)

class BanditModel:
    def __init__(self, train_dir, test_dir, num_gpus, workers_per_gpu, logger, writer, mem_per_cls=50, lr=0.1,
                 max_iter=1000,
                 tr_epoch=256, batch_size=16, sample_lr=0.1):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.cls_list = os.listdir(train_dir)
        self.num_gpus = num_gpus
        self.workers_per_gpu = workers_per_gpu
        self.num_workers = num_gpus * workers_per_gpu
        self.n_classes = len(self.cls_list)
        self.logger = logger
        self.writer = writer

        self.lr = lr
        self.sample_lr = sample_lr

        self.n_per_cls = []
        for cls_name in self.cls_list:
            self.n_per_cls.append(len(os.listdir(os.path.join(train_dir, cls_name))))
        print("cls_list", self.cls_list)
        self.mem_per_cls = mem_per_cls
        self.preference = []
        for i in range(self.n_classes):
            self.preference.append(np.zeros(self.n_per_cls[i]))
        self.max_iter = max_iter
        self.tr_epoch = tr_epoch
        self.batch_size = batch_size
        self.best_action = None
        self.best_reward = 0

        self.iter = 0
        self.mean_reward = 0
        self.running_iter = 0

    def generate_action(self):
        action = []
        for i in range(self.n_classes):
            probs = np.exp(self.preference[i]) / np.sum(np.exp(self.preference[i]))
            action.append(np.random.choice(self.n_per_cls[i], size=self.mem_per_cls, p=probs))
        return action

    def update_model(self, action, reward, worker_id):
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_action = action
        self.mean_reward = self.mean_reward * self.iter / (self.iter + 1) + reward * 1 / (self.iter + 1)
        self.iter += 1
        for i in range(self.n_classes):
            probs = np.exp(self.preference[i]) / np.sum(np.exp(self.preference[i]))
            self.preference[i][action[i]] += self.mem_per_cls * self.sample_lr * (reward - self.mean_reward) * (1 - probs[action[i]])
            non_mask = np.ones(len(self.preference[i]), dtype=bool)
            non_mask[action[i]] = False
            self.preference[i][non_mask] -= self.mem_per_cls * self.sample_lr * (reward - self.mean_reward) * probs[non_mask]
        self.logger.info(f'iteration:{self.iter}/{self.max_iter}, worker:{worker_id}, accuracy:{reward}')
        self.writer.add_scalar('acc', reward, self.iter)

    def learn(self):
        while self.iter < self.max_iter:
            actions = [self.generate_action() for i in range(self.num_workers)]
            workers = [RemoteTrainer.remote(self.n_classes, self.lr, self.train_dir, self.train_dir, self.batch_size,
                                           self.tr_epoch) for i in range(self.num_workers)]
            rewards = ray.get([workers[i].train_worker.remote(i, actions[i], 0) for i in range(self.num_workers)])
            for i in range(self.num_workers):
                self.update_model(actions[i], rewards[i].item(), i)
        optimal_action = []
        for i in range(self.n_classes):
            optimal_action.append(np.argsort(self.preference[i])[-self.mem_per_cls:])
        evaluator = TrainerModel(self.n_classes, self.lr, self.train_dir, self.test_dir, self.batch_size,
                                 self.tr_epoch)
        optimal_reward = evaluator.train_worker(optimal_action, 0)
        self.best_reward = evaluator.train_worker(self.best_action, 0)
        return self.best_action, self.best_reward.item(), optimal_action, optimal_reward.item()


class TrainerModel:
    def __init__(self, n_classes, lr, train_dir, test_dir, batch_size, tr_epoch):
        self.lr = lr
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.tr_epoch = tr_epoch
        self.model_opt = edict({
            'depth': 18,
            'num_classes': n_classes,
            'in_channels': 3,
            'bn': True,
            'normtype': 'BatchNorm',
            'activetype': 'ReLU',
            'pooltype': 'MaxPool2d',
            'preact': False,
            'affine_bn': True,
            'bn_eps': 1e-6,
            'compression': 0.5,
        })
        mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar10')
        self.train_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def train_worker(self, action, device):
        model = ResNet(self.model_opt)
        model = model.cuda(device)
        optimizer, scheduler = select_optimizer('sgd', self.lr, model, 'cos')
        criterion = nn.CrossEntropyLoss(reduction='mean').cuda(device)
        train_dataset = SampleDataset(action, root_dir=self.train_dir, transform=self.train_transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4)
        model.train()
        for epoch in range(self.tr_epoch):
            print(f'epoch:{epoch + 1}')
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr * 0.1
            elif epoch == 1:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr
            else:
                scheduler.step()
            for i, data in enumerate(train_loader):
                x = data['image'].cuda(device)
                y = data['label'].cuda(device)
                optimizer.zero_grad()
                if np.random.rand() < 0.5:
                    x, labels_a, labels_b, lam = cutmix_data(x=x,
                                                             y=y,
                                                             alpha=1.0)
                    logit = model(x)
                    loss = lam * criterion(logit, labels_a) + (
                            1 - lam) * criterion(logit, labels_b)
                else:
                    logit = model(x)
                    loss = criterion(logit, y)
                loss.backward()
                optimizer.step()
        test_dataset = TestDataset(root_dir=self.test_dir, transform=self.test_transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=1)
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data['image'].cuda(device)
                y = data['label'].cuda(device)
                logit = model(x)
                pred = torch.argmax(logit, dim=-1)
                total += logit.size(0)
                correct += torch.sum(y == pred)
        return torch.true_divide(correct, total)

@ray.remote(num_gpus=1 / args.workers_per_gpu)
class RemoteTrainer:
    def __init__(self, n_classes, lr, train_dir, test_dir, batch_size, tr_epoch):
        self.lr = lr
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.tr_epoch = tr_epoch
        self.model_opt = edict({
            'depth': 18,
            'num_classes': n_classes,
            'in_channels': 3,
            'bn': True,
            'normtype': 'BatchNorm',
            'activetype': 'ReLU',
            'pooltype': 'MaxPool2d',
            'preact': False,
            'affine_bn': True,
            'bn_eps': 1e-6,
            'compression': 0.5,
        })
        mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar10')
        self.train_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def train_worker(self, idx, action, device):
        model = ResNet(self.model_opt)
        model = model.cuda(device)
        optimizer, scheduler = select_optimizer('sgd', self.lr, model, 'cos')
        criterion = nn.CrossEntropyLoss(reduction='mean').cuda(device)
        train_dataset = SampleDataset(action, root_dir=self.train_dir, transform=self.train_transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4)
        model.train()
        for epoch in range(self.tr_epoch):
            print(f'worker:{idx}, epoch:{epoch + 1}')
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr * 0.1
            elif epoch == 1:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr
            else:
                scheduler.step()
            for i, data in enumerate(train_loader):
                x = data['image'].cuda(device)
                y = data['label'].cuda(device)
                optimizer.zero_grad()
                if np.random.rand() < 0.5:
                    x, labels_a, labels_b, lam = cutmix_data(x=x,
                                                             y=y,
                                                             alpha=1.0)
                    logit = model(x)
                    loss = lam * criterion(logit, labels_a) + (
                            1 - lam) * criterion(logit, labels_b)
                else:
                    logit = model(x)
                    loss = criterion(logit, y)
                loss.backward()
                optimizer.step()
        test_dataset = TestDataset(root_dir=self.test_dir, transform=self.test_transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=1)
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data['image'].cuda(device)
                y = data['label'].cuda(device)
                logit = model(x)
                pred = torch.argmax(logit, dim=-1)
                total += logit.size(0)
                correct += torch.sum(y == pred)
        return torch.true_divide(correct, total)


class Analyzer:
    def __init__(self, train_dir, batch_size=16):
        self.train_dir = train_dir
        self.cls_list = os.listdir(train_dir)
        self.n_classes = len(self.cls_list)
        self.batch_size = batch_size
        model_opt = edict({
            'depth': 18,
            'num_classes': self.n_classes,
            'in_channels': 3,
            'bn': True,
            'normtype': 'BatchNorm',
            'activetype': 'ReLU',
            'pooltype': 'MaxPool2d',
            'preact': False,
            'affine_bn': True,
            'bn_eps': 1e-6,
            'compression': 0.5,
        })
        self.model = ResNet(model_opt).cuda()
        mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar10')
        self.train_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = TestDataset(root_dir=self.train_dir, transform=self.train_transform)
        self.n_samples = len(train_dataset)
        self.sample_per_cls = self.n_samples//self.n_classes

    def load_model(self, save_path):
        self.model.load_state_dict(torch.load(save_path))

    def train_model(self, lr, tr_epoch, measure=None):
        optimizer, scheduler = select_optimizer('sgd', lr, self.model, 'cos')
        criterion = nn.CrossEntropyLoss(reduction='mean').cuda()
        train_dataset = TestDataset(root_dir=self.train_dir, transform=self.train_transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4)
        self.model.train()
        for epoch in range(tr_epoch):
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * 0.1
            elif epoch == 1:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                scheduler.step()
            for i, data in enumerate(train_loader):
                x = data['image'].cuda()
                y = data['label'].cuda()
                optimizer.zero_grad()
                if np.random.rand() < 0.5:
                    x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                    logit = self.model(x)
                    loss = lam * criterion(logit, labels_a) + (
                            1 - lam) * criterion(logit, labels_b)
                else:
                    logit = self.model(x)
                    loss = criterion(logit, y)
                loss.backward()
                optimizer.step()
            print(f'epoch: {epoch+1}')

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def analyze(self, good_samples, measure):
        if measure == 'uncertainty':
            results = self.uncertainty()
            np.save('results_tsne.npy', results)
        elif measure == 'tsne':
            results = self.tsne()
        if measure == 'uncertainty' or measure == 'tsne':
            good_result = []
            for i, cls_samples in enumerate(good_samples):
                for sample in cls_samples:
                    good_result.append(results[i*self.sample_per_cls+sample])
            np.save('good_results_tsne.npy', good_result)
        elif measure == 'perclass':
            test_dataset = TestDataset(root_dir='./dataset/cifar10/test', transform=self.test_transform)
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=1)
            self.model.eval()
            per_sample_acc = {}
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    x = data['image'].cuda()
                    y = data['label'].cuda()
                    logit = self.model(x)
                    pred = torch.argmax(logit, dim=-1)
                    for j, label in enumerate(y):
                        if label.item() in per_sample_acc:
                            per_sample_acc[label.item()] += (label == pred[j])
                        else:
                            per_sample_acc[label.item()] = int(label == pred[j])
            for i in per_sample_acc:
                print(per_sample_acc[i])
            print(per_sample_acc)

    def likelihood(self, x, y):
        raise NotImplementedError

    def uncertainty(self):
        transform_cands = [CIFAR10Policy()] * 12
        n_transform = len(transform_cands)
        pred_result = np.zeros([self.n_samples, self.n_classes])

        for tr in transform_cands:
            _tr = transforms.Compose([tr] + self.test_transform.transforms)
            test_dataset = TestDataset(root_dir=self.train_dir, transform=_tr)
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=1)
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    x = data['image'].cuda()
                    logit = self.model(x)
                    inference = torch.argmax(logit, dim=1)
                    for idx, pred in enumerate(inference):
                        pred_result[i*self.batch_size+idx, pred] += 1
        return np.max(pred_result, axis=1)/n_transform

    def tsne(self):
        test_dataset = TestDataset(root_dir=self.train_dir, transform=self.test_transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=1)
        self.model.eval()
        features = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data['image'].cuda()
                _, feature = self.model(x, get_feature=True)
                print("feature shape", feature.shape)
                features.append(feature)
        features = torch.cat(features, dim=0).cpu().numpy()
        tsne_features = TSNE(verbose=3, n_iter=50000).fit_transform(features)
        return tsne_features
