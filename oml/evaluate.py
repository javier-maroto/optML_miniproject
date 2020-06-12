import os
from multiprocessing import Lock

import numpy as np
import pandas as pd

from oml.alignment import training_angle_alignment
from oml.angles import quaternion2euler
from tensorflow.keras.optimizers import Ftrl

LOCK = Lock()

def evaluate_and_save_performance(seed, optimizer):
    # Setup
    quaternion_predicted = np.load("data/predicted_quaternions2.npy")
    angles_predicted = quaternion2euler(quaternion_predicted)
    angles_true = np.load("data/angles_true.npy")
    NUM_PROJECTIONS = len(angles_true)

    kwargs = {
        'm': [1.0, 1.0, 1.0, 1.0],
        'steps': 1000,
        'batch_size': 256,
        'projection_idx': range(NUM_PROJECTIONS),
        'angles_true': angles_true,
        'angles_predicted': angles_predicted
    }

    optimizer = optimizer(learning_rate=0.1)
    opt_name = optimizer.__class__.__name__
    # Optimization algorithm
    rotation, loss, collect_data, trajectory = training_angle_alignment(
        optimizer=optimizer,
        seed=seed,
        **kwargs)

    # Save loss
    loss_file_path = f'results/alignment/loss_{opt_name}.csv'
    if not os.path.exists(loss_file_path):
        df = pd.DataFrame(columns=['seed', 'final_loss'])
    else:
        df = pd.read_csv(loss_file_path)
    df = df.append({'seed': seed, 'final_loss': loss}, ignore_index=True)
    LOCK.acquire()
    df.to_csv(loss_file_path, index=False)
    print(f"\t{opt_name} ROC: {(df.final_loss.values < 1).mean() * 100:.2f} %")  # ROC: rate of convergence
    LOCK.release()
