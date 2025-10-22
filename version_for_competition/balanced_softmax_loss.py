import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from typing import Optional


def get_class_frequencies_from_dataset(dataset: Dataset) -> torch.Tensor:
    """Return per-class sample counts (CPU tensor, dtype=int64). Works for ImageFolder.
    Tries .targets or .labels; falls back to iterating .samples if needed.
    """
    # Prefer fast path
    labels = None
    if hasattr(dataset, 'targets') and dataset.targets is not None:
        labels = dataset.targets
    elif hasattr(dataset, 'labels') and dataset.labels is not None:
        labels = dataset.labels
    elif hasattr(dataset, 'samples'):
        # list of (path, label)
        labels = [lbl for _, lbl in dataset.samples]

    if labels is None:
        raise ValueError("Dataset does not expose labels via .targets/.labels/.samples.")

    labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
    num_classes = int(labels_tensor.max().item()) + 1
    freq = torch.bincount(labels_tensor, minlength=num_classes)
    return freq


class BalancedSoftmaxLoss(nn.Module):
    r"""
    Balanced Softmax (NeurIPS 2020) implemented as CE over logits shifted by log(class_count):
        L = CE( logits + log(N_c), y )
    where N_c is the sample count for class c. Using counts (not normalized priors) is equivalent up to an additive constant.

    Notes:
    - Supports `reduction` override at call time and label smoothing.
    - For per-sample losses (sorting), call with reduction='none'.
    """
    def __init__(self, class_frequencies: torch.Tensor, reduction: str = 'mean', tau: float = 1.0):
        super().__init__()
        cf = class_frequencies.float().clamp_min(1.0)
        # store log-counts as buffer so it moves with device
        self.register_buffer('log_counts', cf.log())
        self.reduction = reduction
        self.tau = tau  # optional temperature

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        reduction: Optional[str] = None,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        red = self.reduction if reduction is None else reduction
        # adjust logits: balanced softmax
        log_counts = self.log_counts.to(logits.device)
        logits_adj = logits / self.tau + log_counts  # temperature before shift (common practice)
        # Cross-Entropy handles reduction & smoothing
        return F.cross_entropy(logits_adj, labels, reduction=red, label_smoothing=label_smoothing)

