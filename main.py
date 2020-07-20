#!/usr/bin/env python
# coding: utf-8

# # Main Ideas
# 
# - 4 Classes
# - GroupKFold splitting
# - Class Balance
# - Flips
# - Label Smoothing
# - EfficientNetB2
# - ReduceLROnPlateau


from glob import glob
from collections import OrderedDict
from sklearn.model_selection import GroupKFold
import cv2
import argparse
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn
from sklearn import metrics
import warnings
from efficientnet_pytorch import EfficientNet
from catalyst.data.sampler import BalanceClassSampler, DistributedSamplerWrapper
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
SEED = 42
seed_everything(SEED)
DATA_ROOT_PATH = os.getenv('data_path', '/media/vlad/hdd4_tb/datasets/alaska2')
base_dir = os.getenv("model_save_dir", './')
MODEL_NAME = os.getenv('model_path', 'efficientnet-b7')
MODEL_ARCHITECTURE = os.getenv('model_architecture', 'efficientnet-b7')
NORMALIZATION = os.getenv('norm', 'wo_imagenet')
CUSTOM_CHECKPOINT = os.getenv('custom_checkpoint', 'True')
CUSTOM_CHECKPOINT = True if CUSTOM_CHECKPOINT == 'True' else False
ONLY_SUBMIT = os.getenv('only_submit', 'True')
ONLY_SUBMIT = True if ONLY_SUBMIT == 'True' else False

TRAIN_ON_VALID = os.getenv('train_on_valid', 'False')
TRAIN_ON_VALID = True if TRAIN_ON_VALID == 'True' else False

if 'model_path' in os.environ:
    MODEL_NAME = os.path.join(os.environ['model_path'], os.environ['checkpoint_name'])
    
# model_path = os.path.join(os.environ['model_path'], 'efficientnet-b7-dcc49843.pth')
# b3 1536
# b2 1408
# b7 2560
IN_FEATURES = int(os.getenv('in_features', 2560))

WRITER = SummaryWriter(flush_secs=15, log_dir=os.path.join(base_dir, 'tb_logs'))

class TrainGlobalConfig:
    num_workers = 4
    batch_size = int(os.getenv('batch_size', 25))
    n_epochs = int(os.getenv('epochs', 35))
    lr = float(os.getenv('lr', 0.001))
    grad_accum_steps = int(os.getenv('grad_accum_steps', 0))
    
    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

#     SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
#     scheduler_params = dict(
#         max_lr=0.001,
#         epochs=n_epochs,
#         steps_per_epoch=int(len(train_dataset) / batch_size),
#         pct_start=0.1,
#         anneal_strategy='cos', 
#         final_div_factor=10**5
#     )
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=2,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )

warnings.filterwarnings("ignore")

def run_parallel(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def print_once(msg, end='\n'):
    if (not torch.distributed.is_initialized() or (
            torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):
        print(msg, end=end)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9083'

    # initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()

def get_train_transforms():
    trans = [A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.Resize(height=512, width=512, p=1.0)]
    
    if NORMALIZATION == 'imagenet':
        trans.append(A.augmentations.transforms.Normalize(p=1.0))
    
    trans.append(ToTensorV2(p=1.0))
    return A.Compose(trans, p=1.0)

def get_valid_transforms():
    trans = [A.Resize(height=512, width=512, p=1.0)]

    if NORMALIZATION == 'imagenet':
        trans.append(A.augmentations.transforms.Normalize(p=1.0))
        
    trans.append(ToTensorV2(p=1.0))
    return A.Compose(trans, p=1.0)

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class DatasetRetriever(Dataset):

    def __init__(self, kinds, image_names, labels, transforms=None):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        image = cv2.imread(f'{DATA_ROOT_PATH}/{kind}/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        if NORMALIZATION == 'wo_imagenet':
            image /= 255.0
            
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
            
        target = onehot(4, label)
        return image, target

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
    
    
class DatasetSubmissionRetriever(Dataset):

    def __init__(self, image_names, transforms=None):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        image = cv2.imread(f'{DATA_ROOT_PATH}/Test/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        if NORMALIZATION == 'wo_imagenet':
            image /= 255.0
            
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image_name, image

    def __len__(self) -> int:
        return self.image_names.shape[0]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def alaska_weighted_auc(y_true, y_valid):
    """
    https://www.kaggle.com/anokas/weighted-auc-metric-updated
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        # pdb.set_trace()
        
        if sum(mask) > 0:
            x_padding = np.linspace(fpr[mask][-1], 1, 100)

            x = np.concatenate([fpr[mask], x_padding])
            y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
            y = y - y_min  # normalize such that curve starts at y=0
            score = metrics.auc(x, y)
            submetric = score * weight
            best_subscore = (y_max - y_min) * weight
            competition_metric += submetric

    return competition_metric / normalization
        
class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = alaska_weighted_auc(self.y_true, self.y_pred)
    
    @property
    def avg(self):
        return self.score


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class Fitter:
    
    def __init__(self, model, device, config, base_dir='./', rank=-1):
        self.config = config
        self.epoch = 0
        self.global_step = 0
        
        self.base_dir = base_dir
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device
        self.rank = rank

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.criterion = LabelSmoothing().to(self.device)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, final_scores = self.train_one_epoch(train_loader)

            self.log((f'[RESULT]: Train. Epoch: {self.epoch}, '
                      f'summary_loss: {summary_loss.avg:.5f}, '
                      f'final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}'))
            
            
            t = time.time()
            if self.rank in [-1, 0]:
                self.save(f'{self.base_dir}/last-checkpoint.bin')
                summary_loss_val, final_scores_val = self.validation(validation_loader)

                self.log((f'[RESULT]: Val. Epoch: {self.epoch}, '
                          f'summary_loss: {summary_loss_val.avg:.5f}, '
                          f'final_score: {final_scores_val.avg:.5f}, time: {(time.time() - t):.5f}'))
            
            if self.rank in [-1, 0]:
                if summary_loss_val.avg < self.best_summary_loss:
                    self.best_summary_loss = summary_loss_val.avg
                    self.model.eval()
                    self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                    for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                        os.remove(path)

                    self.save(f'{self.base_dir}/BEST_MODEL.bin')

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1
            
            if self.rank in [-1, 0]:
                WRITER.add_scalars('loss', {'train': summary_loss.avg, 'val': summary_loss_val.avg}, e)
                WRITER.add_scalars('weighted_auc', {'train': final_scores.avg, 'val': final_scores_val.avg}, e)
                WRITER.flush()

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print_once(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)
#             if step == 30:
#                 break

        return summary_loss, final_scores

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        
        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print_once(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            if self.rank in [-1, 0]:
                WRITER.add_scalar('batch_loss', summary_loss.avg, self.global_step)
                WRITER.add_scalar('batch_weighted_auc', final_scores.avg, self.global_step)
                WRITER.flush()
            
            self.global_step += 1
            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            if self.config.grad_accum_steps > 1:
                loss /= self.config.grad_accum_steps
                loss.backward()
                           
                if (step + 1) % self.config.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)

            

            if self.config.step_scheduler:
                self.scheduler.step()
            
#             if step == 30:
#                 break

        return summary_loss, final_scores
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print_once(message)
            
        if self.rank in [0, -1]:
            with open(self.log_path, 'a+') as logger:
                logger.write(f'{message}\n')

def get_net(model_name, in_features, model_architecture):
    if os.path.exists(os.path.dirname(model_name)):
        net = EfficientNet.from_name(model_architecture)
#         net.load_state_dict(torch.load(model_name))
    else:
        net = EfficientNet.from_pretrained(model_architecture)
    net._fc = nn.Linear(in_features=in_features, out_features=4, bias=True)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    return net


def main(rank=-1, world_size=-1):
    # # GroupKFold splitting
    # I think group splitting by image_name is really important for correct validation in this competition ;) 

    if rank == -1:
        device = torch.device('cuda:0')
    else:
        setup(rank, world_size)
        device = rank
        print(device, rank)
        
    dataset = []

    for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
        for path in glob(f'{DATA_ROOT_PATH}/Cover/*.jpg'):
            dataset.append({
                'kind': kind,
                'image_name': path.split('/')[-1],
                'label': label
            })

    random.shuffle(dataset)
    dataset = pd.DataFrame(dataset)

    gkf = GroupKFold(n_splits=5)

    dataset.loc[:, 'fold'] = 0
    
    for fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
        dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number


    fold_number = 0

    train_dataset = DatasetRetriever(
        kinds=dataset[dataset['fold'] != fold_number].kind.values,
        image_names=dataset[dataset['fold'] != fold_number].image_name.values,
        labels=dataset[dataset['fold'] != fold_number].label.values,
        transforms=get_train_transforms(),
    )

    validation_dataset = DatasetRetriever(
        kinds=dataset[dataset['fold'] == fold_number].kind.values,
        image_names=dataset[dataset['fold'] == fold_number].image_name.values,
        labels=dataset[dataset['fold'] == fold_number].label.values,
        transforms=get_valid_transforms(),
    )
    
    net = get_net(MODEL_NAME, IN_FEATURES, MODEL_ARCHITECTURE).to(device)
    
    
    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig, base_dir=base_dir, rank=rank)
    
    if os.path.exists(os.path.dirname(MODEL_NAME)) and CUSTOM_CHECKPOINT:
        fitter.load(MODEL_NAME)
    
    
    sampler = BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling")
    
    if rank != -1:
        sampler = DistributedSamplerWrapper(sampler, num_replicas=world_size, rank=rank)
        
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=TrainGlobalConfig.batch_size,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )
    
    val_loader = DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
    )
    
    if not ONLY_SUBMIT:
        fitter.model = DDP(fitter.model, device_ids=[rank], output_device=rank)
        fitter.fit(train_loader, val_loader)
    
        file = open(f'{base_dir}/log.txt', 'r')
        for line in file.readlines():
            print_once(line[:-1])
        file.close()

    if rank in [0, -1]:
        net = get_net(MODEL_NAME, IN_FEATURES, MODEL_ARCHITECTURE).to(device)
        print_once('Run inference')
        checkpoint = torch.load(os.path.join(base_dir, 'BEST_MODEL.bin'))

        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
            
        net.load_state_dict(new_state_dict)
        net.to(device)
        
        if TRAIN_ON_VALID:
            print_once('Training on validation part')
            net.train()

            val_loader_for_training = DataLoader(
                validation_dataset,
                sampler=sampler,
                batch_size=TrainGlobalConfig.batch_size,
                pin_memory=False,
                drop_last=True,
                num_workers=TrainGlobalConfig.num_workers)
            fitter = Fitter(model=net, device=device, config=TrainGlobalConfig, base_dir=base_dir, rank=rank)
            fitter.config.n_epoch = 1
            fitter.fit(val_loader_for_training, val_loader)
            
        
        net.eval()

        dataset = DatasetSubmissionRetriever(
            image_names=np.array([path.split('/')[-1] for path in glob(f'{DATA_ROOT_PATH}/Test/*.jpg')]),
            transforms=get_valid_transforms(),
        )

        data_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            drop_last=False,
        )

        result = {'Id': [], 'Label': []}
        for step, (image_names, images) in enumerate(data_loader):
            print(f'{step}/{len(data_loader)}', end='\r')

            y_pred = net(images.to(device))
            y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]

            result['Id'].extend(image_names)
            result['Label'].extend(y_pred)


        submission = pd.DataFrame(result)
        submission.to_csv('b2-focall-loss-continue.csv', index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distrib", action='store_true')
    args = parser.parse_args()

    if args.distrib:
        n_gpus = torch.cuda.device_count()
        print_once(f'Running in distributed {n_gpus}')
        run_parallel(main, n_gpus)
    else:
        print_once(f'Running in NOT distributed')
        main()
