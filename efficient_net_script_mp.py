# import os

# os.environ['HTTP_PROXY'] = "http://prusaivabot:T3!-GE8k@proxyru-rd.huawei.com:8080"
# os.environ['HTTPS_PROXY'] = "http://prusaivabot:T3!-GE8k@proxyru-rd.huawei.com:8080"
# os.environ['CURL_CA_BUNDLE'] = ""

# !pip install efficientnet_pytorch
# !pip install albumentations torchvision

from efficientnet_pytorch import EfficientNet
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,
)
import albumentations
from sklearn import metrics
from tqdm import tqdm
import time
from datetime import timedelta
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9083'

    # initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()

def run_parallel(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def get_orig_imgs(path):
    return os.listdir(path)

def get_negative_examples(path):
    return [os.path.join(path, img) for img in get_orig_imgs(path)]

def get_positive_examples(base_path, base_data_path:str, orig_images: List[str]):
    folders = ['JMiPOD', 'JUNIWARD', 'UERD']    
    positive_images = []
    
    for folder in folders:
        for img in orig_images:
            positive_images.append(os.path.join(base_path, folder, img))
    
    return positive_images

def strong_aug(p=0.5):
    return albumentations.Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: np.array, augs=None):
        self.labels = labels
        self.image_paths = image_paths
        
        self.tfms = transforms.Compose([transforms.Resize(512), 
                                        transforms.ToTensor()])
        if augs is not None:
            self.augs = augs()
        else:
            self.augs = augs
        
    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.image_paths[idx]
        image = plt.imread(img)
        image = Image.fromarray(image).convert('RGB')
        
        if self.augs is not None:
            image = self.augs(image=np.array(image))['image']
        else:
            image = np.array(image)
        
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return {'X': torch.tensor(image), 'Y': label}
    
    def __len__(self):
        return len(self.labels)



class EfficientNetClassifier(torch.nn.Module):
    def __init__(self, efficient_net_name):
        super(EfficientNetClassifier, self).__init__()
        self.backbone = EfficientNet.from_pretrained(efficient_net_name, num_classes=1)
    
    def forward(self, batch: torch.Tensor) -> torch.tensor:
        # 3, 512, 512
        return self.backbone.forward(batch)

def evaluate_model(val_dataloader: DataLoader, classifier: torch.nn.Module, criterion):
    epoch_valid_loss = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            x = batch['X'].to(device)
            labels = batch['Y'].to(device)

            logits = classifier.forward(x)
            loss = criterion(logits, labels.float().unsqueeze(dim=1))
            probs = torch.sigmoid(logits.squeeze())
            
            epoch_valid_loss.append(loss.item())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
            
    loss = np.mean(epoch_valid_loss)
    return all_probs, all_labels, loss


def weighted_auc(labels: List[int], preds: List[float], plot = False):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    
    # Calculating ROC curve
    fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label=1)
    # data labels, preds
    area = np.array(tpr_thresholds)[1:] - np.array(tpr_thresholds)[:-1]
    area_normalized = np.dot(area, np.array(weights).T)  # For normalizing AUC
    fscore = 0
    for index, weight in enumerate(weights):
        ymin = tpr_thresholds[index]    
        ymax = tpr_thresholds[index + 1]

        mask = (tpr > ymin) & (tpr < ymax)
        try:
            x = np.concatenate([fpr[mask], np.linspace(fpr[mask][-1], 1, 100)])
        except Exception as e:
            print(fpr)
            print(mask)
            raise e
        y = np.concatenate([tpr[mask], [ymax] * 100])
        y = y #(taking y as origin)
        score = metrics.auc(x, y-ymin)
        # Multiply score with weight
        weighted_score = score * weight

        fscore += weighted_score
        color = ["red", "green"]
        label = ["x ∈ [0, 1], y ∈ [0, 0.4]", "x ∈ [0, 1], y ∈ [0.4, 1.0]"]
        
        if plot:
            plt.title("Separate plots for x ∈ [0, 1], y ∈ [0, 0.4] and x ∈ [0, 1], y ∈ [0.4, 1.0]")
            plt.plot(x, y, color = color[index], label = label[index])
            plt.xlabel("False Positive rate")
            plt.ylabel("True Positive rate")
            plt.legend(loc = 2)
#             plt.plot()

    # Normalizing score
    final_score = fscore/area_normalized
    return final_score

        
    return competition_metric / normalization


def make_submission(model: torch.nn.Module, test_data_path: str, batch_size: int, device: torch.device) -> pd.DataFrame:
    
    images = get_negative_examples(test_data_path)
    
    dataset = ImageDataset(images, [0] * len(images))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(1)).to(device)
    
    all_probs, all_labels, loss = evaluate_model(dataloader, model, criterion)
    all_preds = (torch.tensor(all_probs, dtype=torch.float32) > 0.5).long()
    
    images_names = get_orig_imgs(test_data_path)
    return pd.DataFrame({'Id': images_names, 'Label': all_preds})


def main(rank=-1, world_size=-1):
    epochs = 10
    lr = 1e-5
    save_path = './saved_models/'
    batch_size = 12
    cover_data_path = '/media/vlad/hdd4_tb/datasets/alaska2/Cover/'
    test_data_path = '/media/vlad/hdd4_tb/datasets/alaska2/Test'
    base_data_path = '/media/vlad/hdd4_tb/datasets/alaska2/'
    log_every = 50
    target_metric = 'weighted_auc'
    model_name = 'efficientnet-b1'
    submission_path = 'submissions/submission.csv'
    
    if rank == -1:
        device = torch.device('cuda:0')
    else:
        setup(rank, world_size)
        device = rank
        print(device, rank)

    orig_imgs = get_orig_imgs(cover_data_path)
    test_imgs = get_negative_examples(test_data_path)

    negatives = get_negative_examples(cover_data_path)
    positives = get_positive_examples(base_data_path, cover_data_path, orig_imgs)
    train_paths = negatives + positives
    train_labels = [1] * len(positives) + [0] * len(negatives)

    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        train_paths, train_labels, test_size=0.15, random_state=2020)

    train_dataset = ImageDataset(train_paths, train_labels, strong_aug)
    valid_dataset = ImageDataset(valid_paths, valid_labels, strong_aug)
    
    if rank == -1:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)      
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
        
    val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) 
    classifier = EfficientNetClassifier(model_name)
    classifier = classifier.to(device)
    
    if rank != -1:
        classifier = DDP(classifier, device_ids=[rank])
        
    
        
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(1)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    train_history = []
    val_history = []

    for e in range(epochs):
        epoch_train_loss = []
        epoch_valid_loss = []
        print(f'Running epoch: {e}/{epochs-1}')
        full_save_path = os.path.join(save_path, f'model_ep{e}.pt')
        start_time = time.time()
        
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            x = batch['X'].to(device)
            labels = batch['Y'].to(device)

            logits = classifier.forward(x)
            loss = criterion(logits, labels.float().unsqueeze(dim=1))
            loss.backward()
            optimizer.step()


            epoch_train_loss.append(loss.item())

            if idx % log_every == 0 and rank in [-1, 0]:
                elapsed_time = timedelta(seconds=time.time() - start_time)
                start_time = time.time()
                print(f'Epoch: {e}/{epochs-1}')
                print(f'Batch {idx}/{len(train_dataloader)}: bce_loss: {loss}')
                print(f'Elapsed time {elapsed_time}')
                
        
        if rank in [-1, 0]:
            val_probs, val_labels, val_loss = evaluate_model(val_dataloader, classifier, criterion)
            val_preds =  (torch.tensor(val_probs, dtype=torch.float32) > 0.5).long()

            val_weighted_auc = weighted_auc(val_labels, val_probs)
            auc = metrics.roc_auc_score(val_labels, val_preds)
            val_accuracy = metrics.accuracy_score(val_preds, val_labels)
            val_f1_score = metrics.f1_score(val_preds, val_labels)
            val_history.append({'epoch': e, 'loss': val_loss, 'accuracy': val_accuracy, 'auc': auc,
                                'weighted_auc': val_weighted_auc, 'f1_score': val_f1_score, 'save_path': full_save_path})
            
            torch.save(classifier.state_dict(), full_save_path)

            
        train_history.append({'loss': np.mean(epoch_train_loss)})
    
    if rank in [-1, 0]:
        best_epoch = min(val_history, key=lambda hist: hist[target_metric])
        print(f'Best epoch: {best_epoch}')
        best_load_path = best_epoch['save_path']

        classifier = EfficientNetClassifier(model_name).to(device)
        classifier.load_state_dict(torch.load(best_load_path))

        df = make_submission(classifier, test_data_path, batch_size, device)
        df.to_csv(submission_path)

    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    run_parallel(main, n_gpus)



