# --- START OF FILE: main.py ---

# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from PIL import ImageFile

# ========== 可选导入 timm ==========
try:
    import timm
    HAS_TIMM = True
except ImportError:
    timm = None
    HAS_TIMM = False
# ==================================

from loss_plm import peer_learning_loss
from lr_scheduler import lr_scheduler
from bcnn import BCNN
from resnet import ResNet50
from balanced_softmax_loss import get_class_frequencies_from_dataset, BalancedSoftmaxLoss

# -----------------------------
# 基础设置
# -----------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
np.random.seed(0)

# 固定分辨率 / 卷积大模型下建议打开，能让 cuDNN 选到更快的算法
torch.backends.cudnn.benchmark = True
# 提升 matmul 吞吐（PyTorch 2.x）
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

os.makedirs("model", exist_ok=True)

# -----------------------------
# 参数
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--drop_rate', type=float, default=0.25)
parser.add_argument('--T_k', type=int, default=10, help='how many epochs for linear drop rate')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--step', type=int, required=True, choices=[1, 2], help='1: 只训分类头；2: 全量微调')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--n_classes', type=int, default=0, help="类别数，如果为0则从数据集中自动检测")
parser.add_argument('--net1', type=str, default='bcnn', help='network for model 1（如 convnext_tiny）')
parser.add_argument('--net2', type=str, default='bcnn', help='network for model 2（如 convnext_tiny）')
parser.add_argument('--use_balanced_softmax', action='store_true', help='Enable Balanced Softmax for agreement set sorting.')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor.')

# 训练/推理策略
parser.add_argument('--workers', type=int, default=max(8, (os.cpu_count() or 10) // 1),
                    help='DataLoader workers，建议≈CPU核数')
parser.add_argument('--prefetch_factor', type=int, default=4)
parser.add_argument('--no_channels_last', action='store_true', help='禁用 channels_last')
parser.add_argument('--no_bf16', action='store_true', help='禁用 bf16，仍使用 fp16')
parser.add_argument('--compile', dest='compile', action='store_true', help='启用 torch.compile')
parser.add_argument('--no_compile', dest='compile', action='store_false')
parser.set_defaults(compile=True)
parser.add_argument('--compile_mode', type=str, default='max-autotune',
                    choices=['default', 'reduce-overhead', 'max-autotune'])
parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数')
parser.add_argument('--warmup_epochs', type=int, default=5)

# 图像/模型细节
parser.add_argument('--img_size', type=int, default=224, help='输入分辨率（ConvNeXt/大多数 timm 模型用 224）')
parser.add_argument('--drop_path_rate', type=float, default=0.1, help='timm 模型的 drop_path_rate（仅在支持时生效）')

args = parser.parse_args()

# -----------------------------
# 设备 & 精度
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_channels_last = not args.no_channels_last
use_bf16 = (not args.no_bf16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

data_dir = args.dataset
learning_rate = args.base_lr
batch_size = args.batch_size
num_epochs = args.epoch
drop_rate = args.drop_rate
T_k = args.T_k
weight_decay = args.weight_decay
warmup_epochs = args.warmup_epochs
img_size = args.img_size
dp_rate = args.drop_path_rate

resume = args.resume
epoch_decay_start = 80  # 保留（cosine 下影响小）

project_name = os.path.basename(data_dir.strip('/\\'))
logfile = f'logfile_{project_name}_peer_{args.net1}_{args.net2}_{str(drop_rate)}.txt'

# -----------------------------
# 工具函数
# -----------------------------
def freeze_backbone_unfreeze_classifier(model: nn.Module):
    """ 先冻结全部，再只解冻分类头（兼容 timm 的 get_classifier 接口） """
    for p in model.parameters():
        p.requires_grad = False

    head = None
    if hasattr(model, "get_classifier"):
        try:
            head = model.get_classifier()
        except Exception:
            head = None

    # 如果 get_classifier() 成功返回一个 nn.Module，则只放开它
    if isinstance(head, nn.Module):
        for p in head.parameters():
            p.requires_grad = True
        return

    # 兜底：按常见命名放开
    for n, p in model.named_parameters():
        if n.endswith(('fc.weight', 'fc.bias')) or 'classifier' in n or 'head' in n:
            p.requires_grad = True

# -----------------------------
# 模型工厂
# -----------------------------
def get_model(net_name, n_classes, pretrained, use_two_step):
    if net_name == 'bcnn':
        return BCNN(n_classes=n_classes, pretrained=pretrained, use_two_step=use_two_step)
    elif net_name == 'resnet50':
        return ResNet50(n_classes=n_classes, pretrained=pretrained, use_two_step=use_two_step)
    else:
        if not HAS_TIMM:
            raise ImportError(
                f"Model '{net_name}' 需要 timm，但当前环境未安装。请先执行：pip install timm "
                f"或改用 --net1/--net2 resnet50/bcnn"
            )
        print(f"===> Creating model '{net_name}' from timm library.")
        # 某些 timm 模型不接受 drop_path_rate，做兼容
        try:
            model = timm.create_model(
                net_name,
                pretrained=pretrained,
                num_classes=n_classes,
                drop_path_rate=dp_rate
            )
        except TypeError:
            model = timm.create_model(
                net_name,
                pretrained=pretrained,
                num_classes=n_classes
            )
        # Step 1：只训练分类头
        if pretrained and use_two_step:
            print(f"===> Step 1: Freezing backbone & unfreezing classifier for '{net_name}'")
            freeze_backbone_unfreeze_classifier(model)
        return model

def maybe_compile(m):
    if args.compile and hasattr(torch, "compile"):
        try:
            m = torch.compile(m, mode=args.compile_mode)
        except Exception as e:
            print(f"[WARN] torch.compile 失败，回退原始模型：{e}")
    return m

def wrap_to_device(m):
    # 仅多卡时使用 DP；单卡避免额外开销
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        m = nn.DataParallel(m).to(device)
    else:
        m = m.to(device)
    if use_channels_last:
        m.to(memory_format=torch.channels_last)
    return m

def classifier_params(module_or_dp):
    mod = module_or_dp.module if isinstance(module_or_dp, nn.DataParallel) else module_or_dp
    if hasattr(mod, "get_classifier"):
        head = mod.get_classifier()
        if isinstance(head, nn.Module):
            return head.parameters()
    if hasattr(mod, "fc"):
        return mod.fc.parameters()
    if hasattr(mod, "classifier"):
        return mod.classifier.parameters()
    # 兜底：返回所有参数（避免 0 可训练参数的极端情况）
    return mod.parameters()

# -----------------------------
# 数据
# -----------------------------
print(f"===> Loading data from split folders in: {data_dir}")
train_data_path = os.path.join(data_dir, 'train_split')
val_data_path = os.path.join(data_dir, 'val_split')

if not os.path.isdir(train_data_path) or not os.path.isdir(val_data_path):
    raise FileNotFoundError(f"Error: '{train_data_path}' or '{val_data_path}' not found.\nPlease run the data split script first.")

# 兼容老版 torchvision 无 TrivialAugmentWide
TAW = getattr(torchvision.transforms, "TrivialAugmentWide", None)
train_aug = TAW() if TAW is not None else torchvision.transforms.AutoAugment()

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(size=img_size, scale=(0.7, 1.0)),
    train_aug,
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=img_size),
    torchvision.transforms.CenterCrop(size=img_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
])

train_data = torchvision.datasets.ImageFolder(train_data_path, transform=train_transform)
test_data  = torchvision.datasets.ImageFolder(val_data_path,   transform=test_transform)

N_CLASSES = len(train_data.classes)
if args.n_classes != 0 and args.n_classes != N_CLASSES:
    print(f"Warning: --n_classes ({args.n_classes}) conflicts with detected classes ({N_CLASSES}). Using detected value.")
print(f"===> Automatically detected {N_CLASSES} classes.")

balanced_loss_fn = None
if args.use_balanced_softmax:
    class_freq = get_class_frequencies_from_dataset(train_data)
    balanced_loss_fn = BalancedSoftmaxLoss(class_frequencies=class_freq)
    print("===> Balanced Softmax Loss is ENABLED.")

# DataLoader（高吞吐）
def make_loader(ds, train=True):
    return DataLoader(
        ds,
        batch_size=batch_size if train else batch_size * 2,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        drop_last=train  # 保持 batch 尺寸一致，利于 AMP/compile
    )

train_loader = make_loader(train_data, train=True)
test_loader  = make_loader(test_data,  train=False)

# -----------------------------
# 学习率计划
# -----------------------------
alpha_plan = lr_scheduler(learning_rate, num_epochs, warmup_end_epoch=warmup_epochs, mode='cosine')

def adjust_learning_rate(optimizer, epoch):
    lr = float(alpha_plan[epoch])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(logit, target, topk=(1,)):
    with torch.no_grad():
        # 直接对 logits topk 与 softmax 后等价，用 logits 更高效
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
# 训练与评测
# -----------------------------
def autocast_cm(dtype):
    """ 兼容不同 PyTorch 版本的 autocast """
    if hasattr(torch, "autocast"):
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

def train_one_epoch(train_loader, epoch, model1, optim1, scaler1, model2, optim2, scaler2):
    model1.train()
    model2.train()
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
                label_smoothing=args.label_smoothing
            )
            # 梯度累积
            loss_1 = loss_1 / accum
            loss_2 = loss_2 / accum

        scaler1.scale(loss_1).backward()
        scaler2.scale(loss_2).backward()

        if (it + 1) % accum == 0:
            scaler1.step(optim1)
            scaler2.step(optim2)
            scaler1.update()
            scaler2.update()
            optim1.zero_grad(set_to_none=True)
            optim2.zero_grad(set_to_none=True)

        # 统计
        with torch.no_grad():
            train_total1 += labels.size(0)
            train_correct1 += (logits1.argmax(dim=1) == labels).sum().item()
            train_total2 += labels.size(0)
            train_correct2 += (logits2.argmax(dim=1) == labels).sum().item()

        if (it + 1) % 50 == 0:
            print('Epoch:[{0:03d}/{1:03d}]  Iter:[{2:04d}/{3:04d}]  '
                  'Loss1:[{4:6.4f}]  Loss2:[{5:6.4f}]  '
                  'Iter Runtime:[{6:4.2f}]s'.format(
                   epoch + 1, num_epochs, it + 1, len(train_loader),
                   loss_1.item() * accum, loss_2.item() * accum, time.time() - iter_start_time))

    # 如果最后一次没有对齐累积步，也要 step 一次
    leftover = len(train_loader) % accum
    if leftover != 0:
        scaler1.step(optim1); scaler2.step(optim2)
        scaler1.update(); scaler2.update()
        optim1.zero_grad(set_to_none=True); optim2.zero_grad(set_to_none=True)

    train_acc1 = 100.0 * float(train_correct1) / float(train_total1)
    train_acc2 = 100.0 * float(train_correct2) / float(train_total2)
    return train_acc1, train_acc2

@torch.no_grad()
def evaluate(test_loader, model1, model2):
    model1.eval()
    model2.eval()
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

# -----------------------------
# 训练主程
# -----------------------------
rate_schedule = np.ones(num_epochs, dtype=np.float32) * drop_rate
rate_schedule[:T_k] = np.linspace(0, drop_rate, T_k, dtype=np.float32)

def build_optimizer(params, lr):
    # fused AdamW（2.x）更快；无则回退
    try:
        opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
    except TypeError:
        opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return opt

def count_trainable_params(m: nn.Module):
    return sum(p.requires_grad for p in m.parameters())

def main():
    step = args.step
    print('===> About training in a two-step process! ===')

    if step == 1:
        print('===> Step 1: Training classifier head...')
        m1 = get_model(args.net1, N_CLASSES, pretrained=True,  use_two_step=True)
        m2 = get_model(args.net2, N_CLASSES, pretrained=True,  use_two_step=True)
        m1 = maybe_compile(m1); m2 = maybe_compile(m2)
        m1 = wrap_to_device(m1); m2 = wrap_to_device(m2)
        # 自检：确保有可训练参数
        mod1 = m1.module if isinstance(m1, nn.DataParallel) else m1
        mod2 = m2.module if isinstance(m2, nn.DataParallel) else m2
        assert count_trainable_params(mod1) > 0, "Step 1: Model 1 没有可训练参数，请检查冻结逻辑。"
        assert count_trainable_params(mod2) > 0, "Step 1: Model 2 没有可训练参数，请检查冻结逻辑。"
        optimizer1 = build_optimizer(classifier_params(m1), lr=learning_rate)
        optimizer2 = build_optimizer(classifier_params(m2), lr=learning_rate)

    elif step == 2:
        print('===> Step 2: Fine-tuning whole network...')
        # 默认使用 ImageNet-1k 预训练权重；若 --resume 成功，会覆盖为 Step1 最优
        m1 = get_model(args.net1, N_CLASSES, pretrained=True, use_two_step=False)
        m2 = get_model(args.net2, N_CLASSES, pretrained=True, use_two_step=False)
        m1 = maybe_compile(m1); m2 = maybe_compile(m2)
        m1 = wrap_to_device(m1); m2 = wrap_to_device(m2)
        optimizer1 = build_optimizer(m1.parameters(), lr=learning_rate)
        optimizer2 = build_optimizer(m2.parameters(), lr=learning_rate)
    else:
        raise ValueError('Wrong step argument. Must be 1 or 2.')

    # 仅 fp16 需要 scaler；bf16 不启用
    scaler1 = GradScaler(enabled=(amp_dtype == torch.float16 and device.type == "cuda"))
    scaler2 = GradScaler(enabled=(amp_dtype == torch.float16 and device.type == "cuda"))

    start_epoch = 0
    best_accuracy1 = best_accuracy2 = 0.0
    best_epoch1 = best_epoch2 = 0

    if resume and step == 2:
        print('Resuming from Step 1 best models...')
        model1_path = f'model/net1_step1_{args.net1}_{N_CLASSES}cls_best.pth'
        model2_path = f'model/net2_step1_{args.net2}_{N_CLASSES}cls_best.pth'
        if os.path.isfile(model1_path):
            m1.load_state_dict(torch.load(model1_path, map_location='cpu'), strict=False)
        else:
            print(f"[WARN] Checkpoint not found for model 1 at {model1_path}")
        if os.path.isfile(model2_path):
            m2.load_state_dict(torch.load(model2_path, map_location='cpu'), strict=False)
        else:
            print(f"[WARN] Checkpoint not found for model 2 at {model2_path}")

    with open(logfile, "a") as f:
        f.write(f'------ Starting Step: {step} ...\n')

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        lr1 = adjust_learning_rate(optimizer1, epoch)
        lr2 = adjust_learning_rate(optimizer2, epoch)

        train_acc1, train_acc2 = train_one_epoch(train_loader, epoch, m1, optimizer1, scaler1, m2, optimizer2, scaler2)
        test_acc1,  test_acc2  = evaluate(test_loader, m1, m2)

        # 保存最好
        if test_acc1 > best_accuracy1:
            best_accuracy1 = test_acc1
            best_epoch1 = epoch + 1
            torch.save(m1.state_dict(), f'model/net1_step{step}_{args.net1}_{N_CLASSES}cls_best.pth')
        if test_acc2 > best_accuracy2:
            best_accuracy2 = test_acc2
            best_epoch2 = epoch + 1
            torch.save(m2.state_dict(), f'model/net2_step{step}_{args.net2}_{N_CLASSES}cls_best.pth')

        print('------\n'
              'Epoch: [{:03d}/{:03d}]  LR1: {:.2e}  LR2: {:.2e}\n'
              'Train Acc1: [{:6.2f}%]  Train Acc2: [{:6.2f}%]\n'
              'Val   Acc1: [{:6.2f}%]  Val   Acc2: [{:6.2f}%]\n'
              'Epoch Runtime: [{:6.2f}s]\n'
              '------'.format(
                   epoch + 1, num_epochs, lr1, lr2,
                   train_acc1, train_acc2, test_acc1, test_acc2,
                   time.time() - epoch_start_time))

        with open(logfile, "a") as f:
            f.write(f'Epoch: [{epoch + 1:03d}/{num_epochs:03d}]  '
                    f'LR1: {lr1:.6e}  LR2: {lr2:.6e}\t'
                    f'Train Acc1: [{train_acc1:6.2f}%]\tTrain Acc2: [{train_acc2:6.2f}%]\t'
                    f'Val Acc1: [{test_acc1:6.2f}%]\tVal Acc2: [{test_acc2:6.2f}%]\n')

    print('******\n'
          'Best Accuracy 1: [{0:6.2f}%], at Epoch [{1:03d}]; '
          'Best Accuracy 2: [{2:6.2f}%], at Epoch [{3:03d}].'
          '\n******'.format(best_accuracy1, best_epoch1, best_accuracy2, best_epoch2))
    with open(logfile, "a") as f:
        f.write('******\n'
                f'Best Accuracy 1: [{best_accuracy1:6.2f}%], at Epoch [{best_epoch1:03d}]; '
                f'Best Accuracy 2: [{best_accuracy2:6.2f}%], at Epoch [{best_epoch2:03d}].\n******\n')

if __name__ == '__main__':
    main()

# --- END OF FILE: main.py ---
