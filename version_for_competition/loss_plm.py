# loss_plm.py
import torch
import torch.nn.functional as F
from typing import Optional

@torch.no_grad()
def _topk_small_indices(loss_vec: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of k smallest losses."""
    k = int(max(0, min(k, loss_vec.numel())))
    if k == 0:
        return loss_vec.new_zeros((0,), dtype=torch.long)
    # torch.topk with largest=False gives smallest values
    vals, idx = torch.topk(loss_vec, k=k, largest=False, sorted=False)
    return idx

def _per_sample_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    balanced_softmax_loss=None,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """Return per-sample loss vector (no reduction)."""
    if balanced_softmax_loss is not None:
        return balanced_softmax_loss(
            logits, labels, reduction='none', label_smoothing=label_smoothing
        )
    else:
        return F.cross_entropy(
            logits, labels, reduction='none', label_smoothing=label_smoothing
        )

def _scalar_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    balanced_softmax_loss=None,
    label_smoothing: float = 0.0,
    tau_eff: Optional[float] = None
) -> torch.Tensor:
    """Return scalar loss (mean). If BS provided, use it (optionally with tau override)."""
    if balanced_softmax_loss is None:
        return F.cross_entropy(logits, labels, reduction='mean', label_smoothing=label_smoothing)

    # 临时覆盖 tau（做退火），完后恢复
    old_tau = getattr(balanced_softmax_loss, 'tau', 1.0)
    if tau_eff is not None:
        setattr(balanced_softmax_loss, 'tau', float(tau_eff))
    try:
        loss = balanced_softmax_loss(
            logits, labels, reduction='mean', label_smoothing=label_smoothing
        )
    finally:
        # 恢复，避免对后续 batch 造成副作用
        setattr(balanced_softmax_loss, 'tau', old_tau)
    return loss

def peer_learning_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    labels: torch.Tensor,
    rate: float,
    *,
    balanced_softmax_loss=None,        # 传入 BalancedSoftmaxLoss 实例或 None
    label_smoothing: float = 0.0,
    # τ 退火设置（可按需调整；默认与 main.py 兼容，无需改动主程）
    bs_tau_min: float = 0.7,
    bs_tau_max: float = 1.0,
    anneal_with_rate: bool = True,
    max_rate_hint: float = 0.25        # 你的默认 drop_rate=0.25；不同就改这个或在调用时传参数
):
    """
    Two-network peer learning with small-loss selection on agreement set.
    - agreement: pick k = round((1-rate) * |A|) smallest losses (each net computes its own per-sample loss)
    - disagreement: 全部保留（与原实现一致）
    - 最终回传：若提供 balanced_softmax_loss，则用 BS-CE；否则用普通 CE。
    - τ 退火：默认随当前 rate 从 bs_tau_max 线性降到 bs_tau_min（rate 达到 max_rate_hint 时到达下限）。
    """
    assert logits1.shape == logits2.shape and logits1.shape[0] == labels.shape[0]
    N = labels.shape[0]
    device = logits1.device

    # 预测与一致/分歧划分
    pred1 = logits1.argmax(dim=1)
    pred2 = logits2.argmax(dim=1)
    agree_mask = pred1.eq(pred2)
    agree_idx = torch.nonzero(agree_mask, as_tuple=False).squeeze(1)
    disagree_idx = torch.nonzero(~agree_mask, as_tuple=False).squeeze(1)

    # 计算 agreement 集上的逐样本损失（各自算各自的）
    if agree_idx.numel() > 0:
        l1_agree = _per_sample_loss(logits1[agree_idx], labels[agree_idx],
                                    balanced_softmax_loss, label_smoothing)
        l2_agree = _per_sample_loss(logits2[agree_idx], labels[agree_idx],
                                    balanced_softmax_loss, label_smoothing)

        remember_ratio = float(max(0.0, min(1.0, 1.0 - rate)))
        k = int(round(remember_ratio * agree_idx.numel()))
        idx_small_1 = _topk_small_indices(l1_agree, k)
        idx_small_2 = _topk_small_indices(l2_agree, k)

        keep_idx_1 = agree_idx[idx_small_1]
        keep_idx_2 = agree_idx[idx_small_2]
    else:
        keep_idx_1 = torch.zeros((0,), dtype=torch.long, device=device)
        keep_idx_2 = torch.zeros((0,), dtype=torch.long, device=device)

    # 分歧样本：全部保留（保持与你原实现一致）
    if disagree_idx.numel() > 0:
        final_idx_1 = torch.unique(torch.cat([keep_idx_1, disagree_idx], dim=0))
        final_idx_2 = torch.unique(torch.cat([keep_idx_2, disagree_idx], dim=0))
    else:
        final_idx_1 = keep_idx_1
        final_idx_2 = keep_idx_2

    # 边界保护：若由于极端设置导致空集，则退回到全量
    if final_idx_1.numel() == 0:
        final_idx_1 = torch.arange(N, device=device)
    if final_idx_2.numel() == 0:
        final_idx_2 = torch.arange(N, device=device)

    # τ 退火（可选）：随当前 rate 从 tau_max -> tau_min 线性递减；到 max_rate_hint 封顶
    tau_eff = None
    if balanced_softmax_loss is not None and anneal_with_rate:
        # progress ∈ [0,1]
        progress = float(min(1.0, max(0.0, rate / max(1e-8, max_rate_hint))))
        tau_eff = bs_tau_max - (bs_tau_max - bs_tau_min) * progress

    # 最终标量损失（==> 关键改动：用 BS-CE 而不是普通 CE）
    loss_1_update = _scalar_loss(
        logits1[final_idx_1], labels[final_idx_1],
        balanced_softmax_loss=balanced_softmax_loss,
        label_smoothing=label_smoothing,
        tau_eff=tau_eff
    )
    loss_2_update = _scalar_loss(
        logits2[final_idx_2], labels[final_idx_2],
        balanced_softmax_loss=balanced_softmax_loss,
        label_smoothing=label_smoothing,
        tau_eff=tau_eff
    )

    return loss_1_update, loss_2_update
