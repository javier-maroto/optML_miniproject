import logging
import os
import numpy as np
import pandas as pd
from oml.angles import quaternion2euler
from oml.alignment import training_angle_alignment


def main_alignment(seed, transposed=True):
    logging.basicConfig(filename='logs/alignment.log', level=logging.INFO)

    quaternion_predicted = np.load("data/predicted_quaternions2.npy")

    angles_predicted = quaternion2euler(quaternion_predicted, transposed)

    angles_true = np.load("data/angles_true.npy")
    NUM_PROJECTIONS = len(angles_true)

    m, rotation, loss = training_angle_alignment(
        m=[1.0, 1.0, 1.0, 1.0],
        steps=1000,
        batch_size=256,
        projection_idx=range(NUM_PROJECTIONS),
        learning_rate=0.01,
        angles_true=angles_true,
        angles_predicted=angles_predicted,
        transposed=False,
        seed=seed)

    logging.info(f"m = {m[0]}")
    logging.info(f"R = {rotation[0].numpy()}")
    logging.info(f"loss = {loss}")

    if not os.path.exists('results/alignment/loss.csv'):
        df = pd.DataFrame(columns=['seed', 'final_loss'])
        df.to_csv('results/alignment/loss.csv', index=False)
    else:
        df = pd.read_csv('results/alignment/loss.csv')
        df = df.append({'seed': seed, 'final_loss': loss}, ignore_index=True)
        df.to_csv('results/alignment/loss.csv', index=False)


if __name__ == "__main__":
    for i in range(100):
        main_alignment(i)
