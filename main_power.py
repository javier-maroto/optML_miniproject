"""Generate trajectories and save plots for different learning rate powers (Ftrl)"""
from pathlib import Path

from oml.evaluate import evaluate_and_save_performance_power
from oml.plot import plot_alignment_power


if __name__ == "__main__":
    # Logging
    lr_power_list = (0.0, -0.1, -0.5, -1, -2)
    for lr in lr_power_list:
        evaluate_and_save_performance_power(10, lr)
    tp2 = Path('results/alignment/trajectories/Ftrl/')
    plot_alignment_power(list(tp2.glob("*_power_10.npy")))
