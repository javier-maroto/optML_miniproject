"""Generate trajectories and save plots for different learning rates (Adagrad and Ftrl)"""
from pathlib import Path

from oml.evaluate import evaluate_and_save_performance_lr
from oml.plot import plot_alignment_lr
from tensorflow.keras.optimizers import Ftrl, Adagrad


if __name__ == "__main__":
    # Adagrad
    lr_list = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1)
    Optimizer = Adagrad
    opt_name = Optimizer.__class__.__name__
    for lr in lr_list:
        evaluate_and_save_performance_lr(10, lr, Optimizer)
    tp2 = Path(f'results/alignment/trajectories/{opt_name}/')
    plot_alignment_lr(list(tp2.glob("*_lr_10.npy")), opt_name)

    # Ftrl
    Optimizer = Ftrl
    opt_name = Optimizer.__class__.__name__
    for lr in lr_list:
        evaluate_and_save_performance_lr(10, lr, Optimizer)
    tp2 = Path(f'results/alignment/trajectories/{opt_name}/')
    plot_alignment_lr(list(tp2.glob("*_lr_10.npy")), opt_name)
