import os
import sys
import math
import time
import shutil
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许尽可能解码截断图

from tqdm import tqdm
import numpy as np
from sklearn.mixture import GaussianMixture  # pip install scikit-learn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets


# ------------------------------ 运行时加速开关 ------------------------------
def enable_runtime_speedups():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")  # torch>=2.0
    except Exception:
        pass

def collate_skip_none(batch):
    # 过滤 __getitem__ 返回的 None
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, labels, paths = zip(*batch)
    return torch.stack(imgs, 0), torch.as_tensor(labels), list(paths)

# ------------------------------ 数据集封装 ------------------------------
class NumericSortedImageFolderWithPaths(datasets.ImageFolder):
    def find_classes(self, directory: str):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        try:
            classes_sorted = sorted(classes, key=lambda x: int(x))
        except Exception:
            classes_sorted = sorted(classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes_sorted)}
        return classes_sorted, class_to_idx

    def __getitem__(self, index):
        # 不再调用 super().__getitem__，自己包 try/except，坏图直接返回 None
        path, label = self.samples[index]
        try:
            sample = self.loader(path)          # PIL 打开
            if self.transform is not None:
                sample = self.transform(sample) # 变换
        except Exception:
            return None  # 交给 collate_fn 过滤
        if self.target_transform is not None:
            label = self.target_transform(label)
        return sample, label, path


# ------------------------------ 复制/链接工具 ------------------------------
def _copy_one(src_dst, mode="auto"):
    src, dst = src_dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "hardlink":
        try:
            if os.path.exists(dst): return
            os.link(src, dst)
            return
        except Exception:
            pass  # 回退
    if mode == "symlink":
        try:
            if os.path.exists(dst): return
            os.symlink(os.path.abspath(src), dst)
            return
        except Exception:
            pass  # 回退
    if mode == "auto":
        # 优先硬链接→软链接→复制
        try:
            if os.path.exists(dst): return
            os.link(src, dst); return
        except Exception:
            try:
                if os.path.exists(dst): return
                os.symlink(os.path.abspath(src), dst); return
            except Exception:
                pass
    # 最终复制
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def save_cleaned_dataset_parallel(clean_paths, output_dir, max_workers=8, link_mode="auto"):
    if os.path.exists(output_dir):
        print(f"正在移除已存在的目录: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for path in clean_paths:
        class_name = os.path.basename(os.path.dirname(path))
        dst_dir = os.path.join(output_dir, class_name)
        dst = os.path.join(dst_dir, os.path.basename(path))
        tasks.append((path, dst))

    work = partial(_copy_one, mode=link_mode)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(work, tasks), total=len(tasks), desc=f"保存到 {output_dir}"))


# ------------------------------ 特征提取 ------------------------------
@torch.no_grad()
def extract_features(model, loader, device, use_amp=True, amp_dtype=torch.bfloat16, channels_last=True):
    model.eval()
    feats, labels_all, paths_all = [], [], []
    autocast_ctx = torch.autocast if hasattr(torch, "autocast") else torch.cuda.amp.autocast

    for batch in tqdm(loader, desc="提取特征中"):
        if batch is None:            # ★ 整批都被过滤（全是坏图）就跳过
            continue
        images, labels, paths = batch

        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)

        if use_amp and device.startswith("cuda"):
            with autocast_ctx(device_type="cuda", dtype=amp_dtype):
                out = model(images)
        else:
            out = model(images)

        out = F.normalize(out, dim=1)
        feats.append(out.to(torch.float32).cpu().numpy())   # ★ 先转 float32 再 numpy
        labels_all.extend(labels.numpy().tolist())
        paths_all.extend(paths)

    feats = np.vstack(feats) if len(feats) else np.empty((0, 2048), dtype=np.float32)
    labels_all = np.array(labels_all, dtype=np.int64)
    return feats, labels_all, paths_all


def build_resnet50_encoder(no_compile=False):
    # 兼容新老 torchvision
    try:
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()  # 直接输出 2048 维特征
    return model


# ------------------------------ 清洗逻辑 ------------------------------
def clean_with_gmm_or_percentile(
    all_features, all_labels, num_classes,
    dynamic_threshold=False, gmm_prob_threshold=0.85, selection_percentile=0.7,
    min_images_for_gmm=10
):
    clean_indices = []

    # class-wise prototypes
    dim = all_features.shape[1]
    prototypes = np.zeros((num_classes, dim), dtype=np.float32)
    for i in range(num_classes):
        mask = (all_labels == i)
        if mask.sum() > 0:
            prototypes[i] = all_features[mask].mean(axis=0)
    # 相似度（与所属类原型的点积，特征已归一化）
    all_sims = np.einsum("nd,nd->n", all_features, prototypes[all_labels])

    if dynamic_threshold:
        print("使用 GMM 动态阈值模式...")
        for i in tqdm(range(num_classes), desc="GMM 动态筛选中"):
            mask = (all_labels == i)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            sims = all_sims[mask].reshape(-1, 1)
            if cnt < min_images_for_gmm:
                # 回退固定比例
                k = int(math.ceil(cnt * selection_percentile))
                keep_local = np.argsort(-sims.squeeze())[:k]
                clean_indices.extend(np.where(mask)[0][keep_local].tolist())
                continue
            gmm = GaussianMixture(
                n_components=2, random_state=0, max_iter=100, reg_covar=1e-5
            ).fit(sims)
            clean_comp = int(np.argmax(gmm.means_))
            probs = gmm.predict_proba(sims)[:, clean_comp]
            sel = probs > gmm_prob_threshold
            clean_indices.extend(np.where(mask)[0][sel].tolist())
    else:
        print("使用固定百分比阈值模式...")
        for i in tqdm(range(num_classes), desc="固定比例筛选中"):
            mask = (all_labels == i)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            sims = all_sims[mask]
            k = int(math.ceil(cnt * selection_percentile))
            keep_local = np.argsort(-sims)[:k]
            clean_indices.extend(np.where(mask)[0][keep_local].tolist())

    return clean_indices


# ------------------------------ 主流程 ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SNS-CL 风格的噪声清洗（仅特征 → 动态/固定阈值），为 RTX 4090 优化"
    )
    # 基本
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据集目录（内部直接包含类别子文件夹：0,1,...,N）。")
    parser.add_argument("--output_dir", type=str, default="cleaned_dataset_fast",
                        help="保存结果与缓存的目录。")

    # 推理/加载
    parser.add_argument("--batch_size", type=int, default=512,
                        help="特征提取 batch_size（4090 可从 512 起步，不够就调小）")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers（建议≈CPU核数的 0.5~1.0）")
    parser.add_argument("--image_size", type=int, default=224, help="中心裁剪尺寸")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "none"],
                        help="混合精度类型（4090 推荐 bf16）")
    parser.add_argument("--compile", action="store_true", help="使用 torch.compile 进一步加速（torch>=2.0）")

    # 清洗阈值
    parser.add_argument("--dynamic_threshold", action="store_true",
                        help="启用 GMM 动态阈值；否则按固定比例。")
    parser.add_argument("--gmm_prob_threshold", type=float, default=0.85,
                        help="[动态] 属于“干净”组件的最小概率")
    parser.add_argument("--selection_percentile", type=float, default=0.7,
                        help="[固定] 每类保留比例 (0~1)")
    parser.add_argument("--min_images_per_class_for_gmm", type=int, default=10,
                        help="[动态] 小于该数量则回退到固定比例")

    # 缓存与保存
    parser.add_argument("--save_cache", action="store_true",
                        help="提特征后保存 npz 缓存（下次可直接复用）")
    parser.add_argument("--load_cache", type=str, default=None,
                        help="直接从该 npz 加载特征/标签/路径，跳过模型推理")
    parser.add_argument("--copy_workers", type=int, default=8,
                        help="保存清洗数据时的并行工作线程数")
    parser.add_argument("--link_mode", type=str, default="auto",
                        choices=["auto", "copy", "hardlink", "symlink"],
                        help="保存方式：auto(硬链→软链→复制) / 强制复制 / 强制硬链 / 强制软链")

    args = parser.parse_args()

    enable_runtime_speedups()

    if not os.path.isdir(args.data_path):
        print(f"错误：找不到数据目录 {args.data_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 变换
    eval_transform = transforms.Compose([
        transforms.Resize(int(args.image_size * 256 / 224)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 数据集/加载器
    dataset = NumericSortedImageFolderWithPaths(root=args.data_path, transform=eval_transform)
    num_classes = len(dataset.classes)
    print(f"检测到 {num_classes} 个类别：{dataset.classes[:5]}{' ...' if num_classes > 5 else ''}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=collate_skip_none,  # ★ 加上这一行
    )   

    # -------- 特征阶段 --------
    cache_path = os.path.join(args.output_dir, "feats_cache.npz")
    if args.load_cache:
        print(f"\n从缓存加载特征: {args.load_cache}")
        data = np.load(args.load_cache, allow_pickle=True)
        all_features = data["feats"]
        all_labels = data["labels"]
        all_paths = data["paths"].tolist()
    else:
        print("\n构建 ResNet50 编码器（预训练，不训练）...")
        model = build_resnet50_encoder()
        if device == "cuda":
            model = model.to(device)
            # channels_last 更友好
            model = model.to(memory_format=torch.channels_last)
            if args.compile and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model)
                    print("已启用 torch.compile")
                except Exception as e:
                    print(f"torch.compile 失败：{e}")

        use_amp = (args.amp_dtype != "none") and (device == "cuda")
        amp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "none": torch.float32
        }[args.amp_dtype]

        t0 = time.time()
        all_features, all_labels, all_paths = extract_features(
            model, loader, device,
            use_amp=use_amp, amp_dtype=amp_dtype, channels_last=True
        )
        t1 = time.time()
        print(f"特征提取完成，用时 {t1 - t0:.1f}s；特征形状: {all_features.shape}")

        if args.save_cache:
            np.savez_compressed(cache_path, feats=all_features, labels=all_labels, paths=np.array(all_paths))
            print(f"已保存缓存: {cache_path}")

    # -------- 清洗阶段 --------
    clean_indices = clean_with_gmm_or_percentile(
        all_features, all_labels, num_classes,
        dynamic_threshold=args.dynamic_threshold,
        gmm_prob_threshold=args.gmm_prob_threshold,
        selection_percentile=args.selection_percentile,
        min_images_for_gmm=args.min_images_per_class_for_gmm
    )
    clean_paths = [all_paths[i] for i in clean_indices]

    # -------- 保存阶段 --------
    cleaned_out = os.path.join(args.output_dir, "cleaned_all")
    print("\n保存清洗结果...")
    save_cleaned_dataset_parallel(
        clean_paths, cleaned_out,
        max_workers=args.copy_workers,
        link_mode=args.link_mode
    )

    # -------- 总结 --------
    print("\n--- 清洗总结 ---")
    print(f"数据源目录: {args.data_path}")
    print(f"检测到类别数: {num_classes}")
    print(f"总样本数: {len(all_paths)}")
    print(f"筛选出的干净样本数: {len(clean_paths)} ({(len(clean_paths) / max(1, len(all_paths))):.2%})")
    print(f"清洗结果已保存至: {cleaned_out}")
    print("------------------")


if __name__ == "__main__":
    main()
