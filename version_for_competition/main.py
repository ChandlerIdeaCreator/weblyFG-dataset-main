# -*- coding: utf-8 -*-
"""
Optimized training script for noisy fine-grained classification with two-network peer learning.
- ConvNeXt (timm) + ImageNet-1K pretrained
- Stage1 head-only warm-up; Stage2 full fine-tune
- EMA, RandomErasing, Progressive Resizing, (optional) Weighted Sampler
- Discriminative LR / (optional) ConvNeXt LLRD (applied only to ConvNeXt)
- Fixed prints (pure f-strings), fixed TTA scaling, AMP grad clipping
- Keeps your checkpoint suffix: *_1.pth
- Step1 also saves an extra alias with `_auto` when --n_classes=0 (to silence run_training.sh warning)
"""

import os
import time
import argparse
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from PIL import ImageFile

# Optional timm import
try:
    import timm
    HAS_TIMM = True
except ImportError:
    timm = None
    HAS_TIMM = False

from loss_plm import peer_learning_loss
from lr_scheduler import lr_scheduler
from bcnn import BCNN
from resnet import ResNet50
from balanced_softmax_loss import get_class_frequencies_from_dataset, BalancedSoftmaxLoss

# -----------------------------
# Basics & perf knobs
# -----------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# TF32 for NVIDIA Ampere+ (A100/3090/4090)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

os.makedirs("model", exist_ok=True)

# -----------------------------
# Argparse
# -----------------------------
parser = argparse.ArgumentParser()
# Data & task
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--n_classes', type=int, default=0, help='如果为0则自动从数据集中检测')
# Optim & schedule
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=1e-4)
# Peer learning & label noise
parser.add_argument('--drop_rate', type=float, default=0.25)
parser.add_argument('--T_k', type=int, default=10, help='线性爬升到 drop_rate 的 epoch 数')
parser.add_argument('--use_balanced_softmax', action='store_true')
parser.add_argument('--label_smoothing', type=float, default=0.1)
# Two-step training
parser.add_argument('--step', type=int, required=True, choices=[1, 2], help='1: 只训分类头；2: 全量微调')
parser.add_argument('--resume', action='store_true', help='Step2 从 Step1 最优权重恢复（若存在）')
parser.add_argument('--net1', type=str, default='convnext_base', help='如 convnext_base / bcnn / resnet50 / timm 名称')
parser.add_argument('--net2', type=str, default='bcnn')
# Throughput & precision
parser.add_argument('--workers', type=int, default=max(8, (os.cpu_count() or 10)//1))
parser.add_argument('--prefetch_factor', type=int, default=4)
parser.add_argument('--no_channels_last', action='store_true')
parser.add_argument('--no_bf16', action='store_true')
parser.add_argument('--compile', dest='compile', action='store_true')
parser.add_argument('--no_compile', dest='compile', action='store_false')
parser.set_defaults(compile=True)
parser.add_argument('--compile_mode', type=str, default='max-autotune', choices=['default','reduce-overhead','max-autotune'])
parser.add_argument('--grad_accum_steps', type=int, default=1)
# Image & model specifics
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--drop_path_rate', type=float, default=0.1)
parser.add_argument('--finetune_img_size', type=int, default=384)   # 建议 bcnn 可改 448（见文末提示）
parser.add_argument('--resize_at_epoch', type=int, default=60, help='到该 epoch（从0计）切到高分辨率')
# EMA & eval
parser.add_argument('--ema_decay', type=float, default=0.9999)
parser.add_argument('--use_tta', action='store_true')
parser.add_argument('--tta_scales', type=str, default='1.0,1.15', help='逗号分隔缩放，如 1.0,1.15')
parser.add_argument('--tta_hflip', action='store_true')
# Sampler
parser.add_argument('--use_weighted_sampler', action='store_true', help='类别不均衡时建议打开')
# LR tricks
parser.add_argument('--use_llrd', action='store_true', help='ConvNeXt 分层学习率衰减（仅对 ConvNeXt 生效）')
parser.add_argument('--bb_lr_mult', type=float, default=0.1, help='不启用 LLRD 时：backbone 学习率乘子')

args = parser.parse_args()

# -----------------------------
# Device & precision
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_channels_last = not args.no_channels_last
use_bf16 = (not args.no_bf16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

# Task config
data_dir = args.dataset
learning_rate = args.base_lr
batch_size = args.batch_size
num_epochs = args.epoch
warmup_epochs = args.warmup_epochs
weight_decay = args.weight_decay

# Peer config
drop_rate = args.drop_rate
T_k = args.T_k

# Image & model
img_size = args.img_size
finetune_img_size = args.finetune_img_size
resize_at_epoch = args.resize_at_epoch

# Logging
project_name = os.path.basename(data_dir.strip('/\\'))
logfile = f'logfile_{project_name}_peer_{args.net1}_{args.net2}_{str(drop_rate)}.txt'

# -----------------------------
# Utils
# -----------------------------
def freeze_backbone_unfreeze_classifier(model: nn.Module):
    """Freeze all params then unfreeze classifier head (timm-compatible)."""
    for p in model.parameters():
        p.requires_grad = False
    head = None
    if hasattr(model, 'get_classifier'):
        try:
            head = model.get_classifier()
        except Exception:
            head = None
    if isinstance(head, nn.Module):
        for p in head.parameters():
            p.requires_grad = True
        return
    # Fallback by name patterns
    for n, p in model.named_parameters():
        if n.endswith(('fc.weight', 'fc.bias')) or 'classifier' in n or 'head' in n:
            p.requires_grad = True

def set_eval_for_frozen_norms(model: nn.Module):
    """Put frozen BN layers into eval to keep pretrained stats intact (for BN backbones)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            req = [p.requires_grad for p in m.parameters(recurse=False)]
            if not req or not any(req):
                m.eval()

def is_convnext_name(name: str) -> bool:
    name = (name or "").lower()
    return "convnext" in name

def get_model(net_name, n_classes, pretrained, use_two_step):
    if net_name == 'bcnn':
        return BCNN(n_classes=n_classes, pretrained=pretrained, use_two_step=use_two_step)
    elif net_name == 'resnet50':
        return ResNet50(n_classes=n_classes, pretrained=pretrained, use_two_step=use_two_step)
    else:
        if not HAS_TIMM:
            raise ImportError(
                f"Model '{net_name}' 需要 timm，但当前环境未安装。请先 `pip install timm` 或改用 --net* resnet50/bcnn"
            )
        print(f"===> Creating timm model: {net_name}")
        try:
            model = timm.create_model(
                net_name,
                pretrained=pretrained,
                num_classes=n_classes,
                drop_path_rate=args.drop_path_rate,
            )
        except TypeError:
            model = timm.create_model(
                net_name,
                pretrained=pretrained,
                num_classes=n_classes,
            )
        if pretrained and use_two_step:
            print(f"===> Step1 freeze backbone & unfreeze classifier for {net_name}")
            freeze_backbone_unfreeze_classifier(model)
            set_eval_for_frozen_norms(model)
        return model

def maybe_compile(m):
    if args.compile and hasattr(torch, 'compile'):
        try:
            m = torch.compile(m, mode=args.compile_mode)
        except Exception as e:
            print(f"[WARN] torch.compile 失败，回退原始模型：{e}")
    return m

def wrap_to_device(m):
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        m = nn.DataParallel(m).to(device)
    else:
        m = m.to(device)
    if use_channels_last:
        m.to(memory_format=torch.channels_last)
    return m

def classifier_params(module_or_dp):
    mod = module_or_dp.module if isinstance(module_or_dp, nn.DataParallel) else module_or_dp
    if hasattr(mod, 'get_classifier'):
        head = mod.get_classifier()
        if isinstance(head, nn.Module):
            return head.parameters()
    if hasattr(mod, 'fc'):
        return mod.fc.parameters()
    if hasattr(mod, 'classifier'):
        return mod.classifier.parameters()
    return mod.parameters()

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        src = model.module if isinstance(model, nn.DataParallel) else model
        self.ema = copy.deepcopy(src).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        src = model.module if isinstance(model, nn.DataParallel) else model
        d = self.decay
        for ema_p, src_p in zip(self.ema.parameters(), src.parameters()):
            ema_p.data.mul_(d).add_(src_p.data, alpha=(1.0 - d))

    def to(self, device):
        self.ema.to(device)
        return self

# -----------------------------
# Data
# -----------------------------
print(f"===> Loading data from split folders in: {data_dir}")
train_data_path = os.path.join(data_dir, 'train_split')
val_data_path = os.path.join(data_dir, 'val_split')
if not (os.path.isdir(train_data_path) and os.path.isdir(val_data_path)):
    raise FileNotFoundError(
        f"Error: '{train_data_path}' or '{val_data_path}' not found. Please run the data split script first.")

# Augs
TAW = getattr(torchvision.transforms, 'TrivialAugmentWide', None)
train_aug = TAW() if TAW is not None else torchvision.transforms.AutoAugment()

def build_transforms(size: int, use_erasing: bool = True):
    t_train = [
        torchvision.transforms.RandomResizedCrop(size=size, scale=(0.7, 1.0)),
        train_aug,
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ]
    if use_erasing:
        t_train.append(torchvision.transforms.RandomErasing(p=0.25, scale=(0.02,0.20), ratio=(0.3,3.3)))
    t_train.append(torchvision.transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)))

    t_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=size),
        torchvision.transforms.CenterCrop(size=size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    return torchvision.transforms.Compose(t_train), t_test

train_transform, test_transform = build_transforms(img_size)

train_data = torchvision.datasets.ImageFolder(train_data_path, transform=train_transform)
test_data  = torchvision.datasets.ImageFolder(val_data_path,   transform=test_transform)

N_CLASSES = len(train_data.classes)
if args.n_classes != 0 and args.n_classes != N_CLASSES:
    print(f"[WARN] --n_classes ({args.n_classes}) != detected ({N_CLASSES}); using detected value.")
print(f"===> Automatically detected {N_CLASSES} classes.")

balanced_loss_fn = None
if args.use_balanced_softmax:
    class_freq = get_class_frequencies_from_dataset(train_data)
    balanced_loss_fn = BalancedSoftmaxLoss(class_frequencies=class_freq)
    print("===> Balanced Softmax Loss is ENABLED.")

def make_loader(ds, train=True):
    sampler = None
    shuffle = train
    if train and args.use_weighted_sampler and hasattr(ds, 'targets'):
        targets = torch.tensor(ds.targets)
        class_count = torch.bincount(targets)
        class_weight = 1.0 / (class_count.float() + 1e-6)
        sample_weight = class_weight[targets]
        sampler = WeightedRandomSampler(weights=sample_weight.double(), num_samples=len(sample_weight), replacement=True)
        shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size if train else max(1, int(batch_size * 1.5)),  # eval 稍大，避免 OOM
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        drop_last=train,
    )

train_loader = make_loader(train_data, train=True)
test_loader  = make_loader(test_data,  train=False)

# -----------------------------
# LR schedule
# -----------------------------
alpha_plan = lr_scheduler(learning_rate, num_epochs, warmup_end_epoch=warmup_epochs, mode='cosine')

def adjust_learning_rate(optimizer, epoch):
    lr = float(alpha_plan[epoch])
    for g in optimizer.param_groups:
        # 若 param_group 有自定义 _base_lr（如 LLRD/判别式 LR），用比例维持相对关系
        if '_base_lr' in g and '_raw_lr' in g:
            scale = lr / g['_base_lr']
            g['lr'] = g['_raw_lr'] * scale
        else:
            g['lr'] = lr
    return lr

def accuracy(logit, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        N = target.size(0)
        _, pred = logit.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
            res.append(correct_k.mul_(100.0 / N))
        return res

# -----------------------------
# Train / Eval
# -----------------------------
def autocast_cm(dtype):
    if hasattr(torch, 'autocast'):
        try:
            return torch.autocast(device_type='cuda', dtype=dtype)
        except TypeError:
            pass
        try:
            return torch.autocast('cuda', dtype=dtype)
        except TypeError:
            pass
    try:
        from torch.cuda.amp import autocast as cuda_autocast
        return cuda_autocast(dtype=dtype)
    except Exception:
        return nullcontext()

def train_one_epoch(train_loader, epoch, model1, optim1, scaler1, model2, optim2, scaler2, ema1=None, ema2=None, max_norm=1.0):
    model1.train(); model2.train()
    train_total1 = train_correct1 = 0
    train_total2 = train_correct2 = 0

    accum = max(1, args.grad_accum_steps)
    optim1.zero_grad(set_to_none=True)
    optim2.zero_grad(set_to_none=True)

    for it, batch in enumerate(train_loader):
        iter_start_time = time.time()
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            images, labels = batch

        if use_channels_last:
            images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        with autocast_cm(amp_dtype):
            logits1 = model1(images)
            logits2 = model2(images)
            loss_1, loss_2 = peer_learning_loss(
                logits1, logits2, labels, rate_schedule[epoch],
                balanced_softmax_loss=balanced_loss_fn,
                label_smoothing=args.label_smoothing,
            )
            loss_1 = loss_1 / accum
            loss_2 = loss_2 / accum

        scaler1.scale(loss_1).backward(); scaler2.scale(loss_2).backward()

        if (it + 1) % accum == 0:
            # AMP-safe clip
            scaler1.unscale_(optim1); scaler2.unscale_(optim2)
            try:
                torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm)
            except Exception:
                pass

            scaler1.step(optim1); scaler2.step(optim2)
            scaler1.update();    scaler2.update()
            if ema1 is not None: ema1.update(model1)
            if ema2 is not None: ema2.update(model2)
            optim1.zero_grad(set_to_none=True); optim2.zero_grad(set_to_none=True)

        with torch.no_grad():
            train_total1 += labels.size(0)
            train_correct1 += (logits1.argmax(dim=1) == labels).sum().item()
            train_total2 += labels.size(0)
            train_correct2 += (logits2.argmax(dim=1) == labels).sum().item()

        if (it + 1) % 50 == 0:
            print(
                f"Epoch:[{epoch+1:03d}/{num_epochs:03d}]  Iter:[{it+1:04d}/{len(train_loader):04d}]  "
                f"Loss1:[{loss_1.item()*accum:6.4f}]  Loss2:[{loss_2.item()*accum:6.4f}]  "
                f"Iter Runtime:[{time.time()-iter_start_time:4.2f}]s"
            )

    leftover = len(train_loader) % accum
    if leftover != 0:
        scaler1.unscale_(optim1); scaler2.unscale_(optim2)
        try:
            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm)
        except Exception:
            pass
        scaler1.step(optim1); scaler2.step(optim2)
        scaler1.update();    scaler2.update()
        if ema1 is not None: ema1.update(model1)
        if ema2 is not None: ema2.update(model2)
        optim1.zero_grad(set_to_none=True); optim2.zero_grad(set_to_none=True)

    train_acc1 = 100.0 * float(train_correct1) / float(train_total1)
    train_acc2 = 100.0 * float(train_correct2) / float(train_total2)
    return train_acc1, train_acc2

@torch.no_grad()
def evaluate(test_loader, model1, model2):
    model1.eval(); model2.eval()
    correct1 = total1 = 0
    correct2 = total2 = 0

    for batch in test_loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            images, labels = batch

        if use_channels_last:
            images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        with autocast_cm(amp_dtype):
            logits1 = model1(images)
            pred1 = logits1.argmax(dim=1)
            total1 += labels.size(0)
            correct1 += (pred1 == labels).sum().item()

            logits2 = model2(images)
            pred2 = logits2.argmax(dim=1)
            total2 += labels.size(0)
            correct2 += (pred2 == labels).sum().item()

    acc1 = 100.0 * float(correct1) / float(total1)
    acc2 = 100.0 * float(correct2) / float(total2)
    return acc1, acc2

@torch.no_grad()
def evaluate_single(model, loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        if use_channels_last:
            images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        with autocast_cm(amp_dtype):
            logits = model(images)
            pred = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100.0 * correct / total

@torch.no_grad()
def evaluate_tta(model, loader, scales=(1.0, 1.15), hflip=True):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits_agg = None
        for s in scales:
            if s != 1.0:
                new_size = int(images.shape[-1] * s)
                imgs = F.interpolate(images, size=(new_size, new_size), mode='bilinear', align_corners=False)
            else:
                imgs = images
            with autocast_cm(amp_dtype):
                out = model(imgs)
                if logits_agg is None:
                    logits_agg = out
                else:
                    logits_agg = logits_agg + out
                if hflip:
                    out_flip = model(torch.flip(imgs, dims=[-1]))
                    logits_agg = logits_agg + out_flip
        pred = logits_agg.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100.0 * correct / total

# -----------------------------
# LR param groups
# -----------------------------
def optim_groups_discriminative(model, base_lr, wd, bb_mult=0.1):
    head_params = list(classifier_params(model))
    head_ids = {id(p) for p in head_params}
    bb_params = [p for p in model.parameters() if id(p) not in head_ids]
    groups = [
        {'params': bb_params, 'lr': base_lr * bb_mult, 'weight_decay': wd, '_raw_lr': base_lr * bb_mult},
        {'params': head_params, 'lr': base_lr,         'weight_decay': wd, '_raw_lr': base_lr},
    ]
    for g in groups: g['_base_lr'] = base_lr
    return groups

def optim_groups_convnext_llrd(model, base_lr, wd, decay=0.7):
    """Layer-wise LR decay for ConvNeXt: later stages larger LR."""
    mod = model.module if isinstance(model, nn.DataParallel) else model
    groups, used = [], set()
    stage_names = ['stages.0','stages.1','stages.2','stages.3']
    L = len(stage_names)
    for i, sn in enumerate(stage_names):
        lr = base_lr * (decay ** (L-1-i))
        params = [p for n,p in mod.named_parameters() if n.startswith(sn)]
        if params:
            groups.append({'params': params, 'lr': lr, 'weight_decay': wd, '_raw_lr': lr})
            used.update(map(id, params))
    # head
    head = mod.get_classifier() if hasattr(mod,'get_classifier') else None
    if isinstance(head, nn.Module):
        hp = list(head.parameters()); used.update(map(id, hp))
        if hp: groups.append({'params': hp, 'lr': base_lr, 'weight_decay': wd, '_raw_lr': base_lr})
    # rest fallback
    rest = [p for p in mod.parameters() if id(p) not in used]
    if rest:
        groups.append({'params': rest, 'lr': base_lr*(decay**L), 'weight_decay': wd, '_raw_lr': base_lr*(decay**L)})
    for g in groups: g['_base_lr'] = base_lr
    return groups

# -----------------------------
# Main
# -----------------------------
rate_schedule = np.ones(num_epochs, dtype=np.float32) * drop_rate
rate_schedule[:T_k] = np.linspace(0, drop_rate, T_k, dtype=np.float32)

def count_trainable_params(m: nn.Module):
    return sum(p.requires_grad for p in m.parameters())

def save_ckpt(path, state_dict):
    torch.save(state_dict, path)

def save_ckpt_best(path, ema_model):
    save_ckpt(path, ema_model.state_dict())

def main():
    print('===> Two-step training (peer learning) starting...')
    step = args.step

    # Build models
    if step == 1:
        print('===> Step 1: Train classifier head only')
        m1 = get_model(args.net1, N_CLASSES, pretrained=True,  use_two_step=True)
        m2 = get_model(args.net2, N_CLASSES, pretrained=True,  use_two_step=True)
        m1 = maybe_compile(m1); m2 = maybe_compile(m2)
        m1 = wrap_to_device(m1); m2 = wrap_to_device(m2)
        assert count_trainable_params(m1.module if isinstance(m1, nn.DataParallel) else m1) > 0, 'Model1 has no trainable params in Step1'
        assert count_trainable_params(m2.module if isinstance(m2, nn.DataParallel) else m2) > 0, 'Model2 has no trainable params in Step1'
        optimizer1 = optim.AdamW(classifier_params(m1), lr=learning_rate, weight_decay=weight_decay)
        optimizer2 = optim.AdamW(classifier_params(m2), lr=learning_rate, weight_decay=weight_decay)
    elif step == 2:
        print('===> Step 2: Fine-tune whole networks')
        m1 = get_model(args.net1, N_CLASSES, pretrained=True,  use_two_step=False)
        m2 = get_model(args.net2, N_CLASSES, pretrained=True,  use_two_step=False)
        m1 = maybe_compile(m1); m2 = maybe_compile(m2)
        m1 = wrap_to_device(m1); m2 = wrap_to_device(m2)

        # LLRD 仅对 ConvNeXt 应用；BCNN/其他骨干走判别式 LR（避免“假 LLRD”退化）
        if args.use_llrd and is_convnext_name(args.net1):
            print("===> net1 uses ConvNeXt LLRD param groups")
            opt_groups1 = optim_groups_convnext_llrd(m1, learning_rate, weight_decay, decay=0.7)
        else:
            print(f"===> net1 uses discriminative LR: backbone x{args.bb_lr_mult}")
            opt_groups1 = optim_groups_discriminative(m1, learning_rate, weight_decay, bb_mult=args.bb_lr_mult)

        if args.use_llrd and is_convnext_name(args.net2):
            print("===> net2 uses ConvNeXt LLRD param groups")
            opt_groups2 = optim_groups_convnext_llrd(m2, learning_rate, weight_decay, decay=0.7)
        else:
            print(f"===> net2 uses discriminative LR: backbone x{args.bb_lr_mult}")
            opt_groups2 = optim_groups_discriminative(m2, learning_rate, weight_decay, bb_mult=args.bb_lr_mult)

        optimizer1 = optim.AdamW(opt_groups1, lr=learning_rate, weight_decay=weight_decay)
        optimizer2 = optim.AdamW(opt_groups2, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError('Wrong step argument. Must be 1 or 2.')

    # Scalers (only for fp16)
    scaler1 = GradScaler(enabled=(amp_dtype == torch.float16 and device.type == 'cuda'))
    scaler2 = GradScaler(enabled=(amp_dtype == torch.float16 and device.type == 'cuda'))

    # EMA wrappers
    ema1 = ModelEMA(m1, decay=args.ema_decay).to(device)
    ema2 = ModelEMA(m2, decay=args.ema_decay).to(device)

    # Resume Step2 from Step1 best (keep *_1.pth) — also try `_auto` alias to match run_training.sh
    if args.resume and step == 2:
        print('===> Resuming from Step1 best checkpoints (if available)')
        p1 = f'model/net1_step1_{args.net1}_{N_CLASSES}cls_best_1.pth'
        p2 = f'model/net2_step1_{args.net2}_{N_CLASSES}cls_best_1.pth'
        p1_auto = f'model/net1_step1_{args.net1}_autocls_best_1.pth'
        p2_auto = f'model/net2_step1_{args.net2}_autocls_best_1.pth'

        loaded1 = False
        if os.path.isfile(p1):
            m1.load_state_dict(torch.load(p1, map_location='cpu'), strict=False); loaded1 = True
        elif os.path.isfile(p1_auto):
            print(f'[INFO] Falling back to auto-named ckpt: {p1_auto}')
            m1.load_state_dict(torch.load(p1_auto, map_location='cpu'), strict=False); loaded1 = True
        else:
            print(f'[WARN] Not found: {p1} or {p1_auto}')

        loaded2 = False
        if os.path.isfile(p2):
            m2.load_state_dict(torch.load(p2, map_location='cpu'), strict=False); loaded2 = True
        elif os.path.isfile(p2_auto):
            print(f'[INFO] Falling back to auto-named ckpt: {p2_auto}')
            m2.load_state_dict(torch.load(p2_auto, map_location='cpu'), strict=False); loaded2 = True
        else:
            print(f'[WARN] Not found: {p2} or {p2_auto}')

    # Log
    with open(logfile, 'a') as f:
        f.write(f'------ Starting Step: {step} ...\n')

    # Parse TTA scales
    try:
        tta_scales = tuple(float(x) for x in args.tta_scales.split(','))
    except Exception:
        tta_scales = (1.0, 1.15)

    best_accuracy1 = best_accuracy2 = 0.0
    best_epoch1 = best_epoch2 = 0

    for epoch in range(num_epochs):
        # Progressive resizing
        if epoch == resize_at_epoch and finetune_img_size > img_size:
            print(f'[INFO] Switch resolution: {img_size} -> {finetune_img_size}')
            new_train_tf, new_test_tf = build_transforms(finetune_img_size)
            train_data.transform = new_train_tf
            test_data.transform  = new_test_tf

        epoch_start = time.time()
        lr1 = adjust_learning_rate(optimizer1, epoch)
        lr2 = adjust_learning_rate(optimizer2, epoch)

        train_acc1, train_acc2 = train_one_epoch(
            train_loader, epoch, m1, optimizer1, scaler1, m2, optimizer2, scaler2, ema1=ema1, ema2=ema2, max_norm=1.0
        )

        # Raw model eval
        test_acc1, test_acc2 = evaluate(test_loader, m1, m2)
        # EMA single-crop eval
        ema_acc1 = evaluate_single(ema1.ema, test_loader)
        ema_acc2 = evaluate_single(ema2.ema, test_loader)
        # Optional TTA on EMA
        if args.use_tta:
            ema_tta1 = evaluate_tta(ema1.ema, test_loader, scales=tta_scales, hflip=args.tta_hflip)
            ema_tta2 = evaluate_tta(ema2.ema, test_loader, scales=tta_scales, hflip=args.tta_hflip)
            metric1, metric2 = ema_tta1, ema_tta2
            tta_line = f"TTA Acc1: [{ema_tta1:6.2f}%]  TTA Acc2: [{ema_tta2:6.2f}%]"
        else:
            metric1, metric2 = ema_acc1, ema_acc2
            tta_line = ""

        # Save best by EMA (or EMA+TTA) metric — keep *_1.pth
        # 另存 `_auto` 别名以配合 run_training.sh 的检查（当 --n_classes=0）
        save_alias = (args.n_classes == 0)

        if metric1 > best_accuracy1:
            best_accuracy1 = metric1; best_epoch1 = epoch + 1
            p_main = f'model/net1_step{step}_{args.net1}_{N_CLASSES}cls_best_1.pth'
            save_ckpt_best(p_main, ema1.ema)
            if save_alias and step == 1:
                p_alias = f'model/net1_step1_{args.net1}_autocls_best_1.pth'
                save_ckpt_best(p_alias, ema1.ema)

        if metric2 > best_accuracy2:
            best_accuracy2 = metric2; best_epoch2 = epoch + 1
            p_main = f'model/net2_step{step}_{args.net2}_{N_CLASSES}cls_best_1.pth'
            save_ckpt_best(p_main, ema2.ema)
            if save_alias and step == 1:
                p_alias = f'model/net2_step1_{args.net2}_autocls_best_1.pth'
                save_ckpt_best(p_alias, ema2.ema)

        # Also save last (optional)
        p_last1 = f'model/net1_step{step}_{args.net1}_{N_CLASSES}cls_last_1.pth'
        p_last2 = f'model/net2_step{step}_{args.net2}_{N_CLASSES}cls_last_1.pth'
        save_ckpt_best(p_last1, ema1.ema)
        save_ckpt_best(p_last2, ema2.ema)
        if save_alias and step == 1:
            save_ckpt_best(f'model/net1_step1_{args.net1}_autocls_last_1.pth', ema1.ema)
            save_ckpt_best(f'model/net2_step1_{args.net2}_autocls_last_1.pth', ema2.ema)

        print(
            f"------\n"
            f"Epoch: [{epoch+1:03d}/{num_epochs:03d}]  LR1: {lr1:.2e}  LR2: {lr2:.2e}\n"
            f"Train Acc1: [{train_acc1:6.2f}%]  Train Acc2: [{train_acc2:6.2f}%]\n"
            f"Val   Acc1 (raw): [{test_acc1:6.2f}%]  Val   Acc2 (raw): [{test_acc2:6.2f}%]\n"
            f"Val   Acc1 (EMA): [{ema_acc1:6.2f}%]  Val   Acc2 (EMA): [{ema_acc2:6.2f}%]\n"
            f"{tta_line}\n"
            f"Epoch Runtime: [{time.time()-epoch_start:6.2f}s]\n"
            f"------"
        )

        with open(logfile, 'a') as f:
            f.write(
                f"Epoch: [{epoch + 1:03d}/{num_epochs:03d}]  "
                f"LR1: {lr1:.6e}  LR2: {lr2:.6e}\t"
                f"Train Acc1: [{train_acc1:6.2f}%]\tTrain Acc2: [{train_acc2:6.2f}%]\t"
                f"Val Acc1 (raw): [{test_acc1:6.2f}%]\tVal Acc2 (raw): [{test_acc2:6.2f}%]\t"
                f"Val Acc1 (EMA): [{ema_acc1:6.2f}%]\tVal Acc2 (EMA): [{ema_acc2:6.2f}%]"
                f"{(f'\tTTA1: [{ema_tta1:6.2f}%]\tTTA2: [{ema_tta2:6.2f}%]' if args.use_tta else '')}\n"
            )

    print(
        f"******\n"
        f"Best (EMA{' +TTA' if args.use_tta else ''}) Accuracy 1: [{best_accuracy1:6.2f}%], at Epoch [{best_epoch1:03d}];\n"
        f"Best (EMA{' +TTA' if args.use_tta else ''}) Accuracy 2: [{best_accuracy2:6.2f}%], at Epoch [{best_epoch2:03d}].\n"
        f"******"
    )

    with open(logfile, 'a') as f:
        f.write(
            "******\n"
            f"Best (EMA{' +TTA' if args.use_tta else ''}) Acc1: [{best_accuracy1:6.2f}%] at Epoch [{best_epoch1:03d}]; "
            f"Best (EMA{' +TTA' if args.use_tta else ''}) Acc2: [{best_accuracy2:6.2f}%] at Epoch [{best_epoch2:03d}].\n"
            "******\n"
        )

if __name__ == '__main__':
    main()
