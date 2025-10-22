# --- START OF FILE: predict.py ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, csv
from typing import List, Optional, Tuple

import torch
import torchvision
from PIL import Image

# --- [NEW] ---
import timm
# ---------------

# 你的仓库里就有这两个文件
from bcnn import BCNN
from resnet import ResNet50


def load_class_names(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    return names if names else None

# --- [MODIFIED] ---
# 修改 build_model 以支持 timm
def build_model(net: str, n_classes: int):
    if net == "bcnn":
        model = BCNN(n_classes=n_classes, pretrained=False)
    elif net == "resnet50":
        model = ResNet50(n_classes=n_classes, pretrained=False)
    else:
        # 使用 timm 加载任何其他模型
        print(f"===> Building model '{net}' from timm library for prediction.")
        model = timm.create_model(net, pretrained=False, num_classes=n_classes)
    return model
# --------------------

def load_weights(model: torch.nn.Module, path: str) -> torch.nn.Module:
    # 兼容 CPU / GPU 权重加载
    map_loc = None if torch.cuda.is_available() else "cpu"
    state = torch.load(path, map_location=map_loc)
    model.load_state_dict(state)
    return model

# --- [MODIFIED] ---
# 将默认尺寸改为 224
def build_transform(img_size: int = 224):
    # 与训练/评测一致的预处理
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225)),
    ])
# --------------------

class ImageFolderFlat(torch.utils.data.Dataset):
    """读取一个目录下的所有图片文件（不要求子目录/标签）"""
    def __init__(self, root: str, tfm, recursive: bool = False,
                 exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
        self.paths = []
        root = os.path.abspath(root)
        if recursive:
            for dp, _, files in os.walk(root):
                for fn in files:
                    if os.path.splitext(fn)[1].lower() in exts:
                        self.paths.append(os.path.join(dp, fn))
        else:
            for fn in os.listdir(root):
                if os.path.splitext(fn)[1].lower() in exts:
                    self.paths.append(os.path.join(root, fn))
        self.paths.sort()
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p)
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return self.tfm(img), p


@torch.no_grad()
def infer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    netA = build_model(args.net, args.n_classes)
    if device == "cuda":
        netA = torch.nn.DataParallel(netA).to(device)
    netA = load_weights(netA, args.model)
    netA.eval()

    netB = None
    if args.model2:
        netB = build_model(args.net2, args.n_classes)
        if device == "cuda":
            netB = torch.nn.DataParallel(netB).to(device)
        netB = load_weights(netB, args.model2)
        netB.eval()

    tfm = build_transform(img_size=args.img_size)
    ds = ImageFolderFlat(args.images, tfm, recursive=args.recursive)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )
    names = load_class_names(args.class_names)

    rows = [("filename", "pred_idx")]
    for x, paths in loader:
        x = x.to(device) if device == "cuda" else x
        logitsA = netA(x)
        if netB is not None:
            logitsB = netB(x)
            logits = (logitsA + logitsB) / 2.0
        else:
            logits = logitsA
        pred = logits.argmax(dim=1).cpu().tolist()

        for pth, cls_idx in zip(paths, pred):
            cls_name = names[cls_idx] if names and 0 <= cls_idx < len(names) else ""
            rows.append((os.path.relpath(pth, args.images), int(cls_idx)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    print(f"Saved predictions to: {args.out}")
    print(f"Total images: {len(ds)}")


def parse_args():
    ap = argparse.ArgumentParser(description="Flat-folder image prediction (single or ensemble)")
    ap.add_argument("--images", type=str, required=True, help="文件夹：包含待预测图片（可配合 --recursive）")
    ap.add_argument("--n_classes", type=int, required=True, help="类别数，需与训练一致")
    # --- [MODIFIED] ---
    # 移除 choices 限制以允许 convnext_base 等
    ap.add_argument("--net", type=str, default="bcnn", help="模型 A 的结构")
    ap.add_argument("--model", type=str, required=True, help="模型 A 的权重 .pth 路径")
    ap.add_argument("--net2", type=str, default="bcnn", help="模型 B 的结构")
    # --------------------
    ap.add_argument("--model2", type=str, default="", help="模型 B 的权重 .pth 路径（留空则不用集成）")
    # --- [MODIFIED] ---
    # 将默认尺寸改为 224
    ap.add_argument("--img_size", type=int, default=224, help="输入尺寸（与训练/评测一致）")
    # --------------------
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--recursive", action="store_true", help="递归遍历子目录中的图片")
    ap.add_argument("--class_names", type=str, default="", help="可选：类别名文本文件（每行一个类名）")
    ap.add_argument("--out", type=str, default="pred.csv", help="输出 CSV 路径")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args)

# --- END OF FILE: predict.py ---