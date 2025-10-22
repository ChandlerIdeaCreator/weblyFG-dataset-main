# lr_scheduler.py
import math
import numpy as np


def lr_warmup(lr_list, lr_init, warmup_end_epoch=5):
    """Linear warmup from 0 to lr_init over [0, warmup_end_epoch-1]."""
    warm = max(0, int(warmup_end_epoch))
    if warm > 0:
        # inclusive 到 lr_init（与主程比例缩放兼容）
        lr_list[:warm] = list(np.linspace(0.0, lr_init, warm, endpoint=True))
    return lr_list


def lr_scheduler(lr_init, num_epochs, warmup_end_epoch=5, mode='cosine', min_lr=0.0):
    """
    Build a length-`num_epochs` LR list.

    Cosine after warmup (exactly reaches `min_lr` at the final epoch):
        For t in [0, T-1],  lr = min_lr + 0.5*(lr_init - min_lr)*(1 + cos(pi * t / (T-1)))
        where T = max(1, num_epochs - warmup_end_epoch)
    """
    num_epochs = int(num_epochs)
    assert num_epochs > 0
    lr_list = [lr_init] * num_epochs

    # Warmup first
    lr_list = lr_warmup(lr_list, lr_init, warmup_end_epoch)

    print(f"*** learning rate warms up for {int(warmup_end_epoch)} epochs (0 .. {max(0, int(warmup_end_epoch)-1)})")
    print(f"*** learning rate decays in {mode} mode")

    if int(warmup_end_epoch) >= num_epochs:
        # 全程都在 warmup，直接返回（避免越界）
        return lr_list

    if mode == 'cosine':
        T = max(1, num_epochs - int(warmup_end_epoch))
        idx0 = int(warmup_end_epoch)
        # 余弦段：i=0..T-1，严格保证最后一个 epoch 到 min_lr
        for i in range(T):
            # 当 i=0 => lr=lr_init；当 i=T-1 => lr=min_lr
            cos_term = 0.5 * (1.0 + math.cos(math.pi * i / max(1, T - 1)))
            lr_list[idx0 + i] = float(min_lr + (lr_init - min_lr) * cos_term)
    else:
        raise AssertionError(f'{mode} mode is not implemented')

    return lr_list


if __name__ == '__main__':
    print('===> Test lr scheduler - cosine mode')
    print(lr_scheduler(0.01, 10, 3))
