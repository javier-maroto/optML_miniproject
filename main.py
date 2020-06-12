"""Generate trajectories and save plots for different optimization algorithms"""
import itertools
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tensorflow.keras.optimizers import (SGD, Adagrad, Adam, Adamax, Ftrl,
                                         Nadam, RMSprop)

from oml.evaluate import evaluate_and_save_performance
from oml.plot import plot_alignment

if __name__ == "__main__":
    # Logging
    logging.basicConfig(filename='logs/alignment.log', level=logging.INFO)
    list_optimizers = (Adam, RMSprop, SGD, Adagrad, Adamax, Ftrl, Nadam)
    tp = Path('results/alignment/trajectories')
    with Pool(processes=len(list_optimizers)) as p:
        for i in np.arange(1, 100):
            # Generate trajectory numpy files
            p.starmap(evaluate_and_save_performance, itertools.product([i], list_optimizers))
            # Save trajectory plots
            plot_alignment(list(tp.glob(f"*/{i}.npy")))
            # Sanity check (all trajectories share initial point for the seed chosen)
            first = None
            for opt in list_optimizers:
                opt_name = opt.__name__
                first2 = np.load(f'results/alignment/trajectories/{opt_name}/{i}.npy')[0, :]
                if first is None:
                    first = first2
                    continue
                assert (first == first2).all()
