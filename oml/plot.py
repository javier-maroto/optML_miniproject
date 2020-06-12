import itertools
import logging
import os
from multiprocessing import Lock, Pool
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import (SGD, Adagrad, Adam, Adamax, Ftrl,
                                         Nadam, RMSprop)

from oml.alignment import loss_alignment, training_angle_alignment
from oml.angles import create_unique_angle, euler2quaternion


def plot_alignment(paths, w_good=50, resolution=(50, 50)):
    # Setup
    q_pred = np.load("data/predicted_quaternions2.npy")
    angles_true = np.load("data/angles_true.npy")
    q_true = euler2quaternion(angles_true)
    good_point = np.load(f'results/alignment/trajectories/Adagrad/1.npy')[-1, :]

    # Load trajectories of all the optimizers
    tt = {}
    seed = int(paths[0].name[:-4])
    save_path = f"results/alignment/trajectories_plots/{seed}.png"
    for path in paths:
        opt = str(path).split('/')[3]
        tt[opt] = np.load(path)

    # Align trajectories: use the unique angle of the starting point as reference (the point is
    #   the same for all the optimizers)
    points = []
    for k, t in tt.items():
        _, fun, diff1, diff2 = create_unique_angle(t[0], ret_full=True)
        t = np.array([[fun[i](tt[i] + diff1[i]) + diff2[i] for i in range(6)] for tt in t])
        points += list(t)
        tt[k] = t
    
    # Add the optimization solution (found beforehand from one converging seed). This forces the 
    #   PCA to take into account this point when plotting the loss curves
    points = np.stack(points + [good_point] * w_good)
    model = PCA(n_components=2)
    p = model.fit_transform(points)
    
    # Sample uniformly from the 2D transformed space (which is what the plot shows)
    ls_dim = lambda dim: np.linspace(p[:, dim].min(), p[:, dim].max(), resolution[dim])
    X, Y = np.meshgrid(ls_dim(0), ls_dim(1))
    Z = np.zeros_like(X)

    # Obtain the loss for each of the sampled points
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            angle = model.inverse_transform((X[i, j], Y[i, j]))
            angle = [tf.Variable(angle)]
            Z[i,j] = loss_alignment([1., 1., 1., 1.], angle, q_pred, q_true)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cp = ax.contourf(X, Y, Z, levels=np.linspace(0, np.pi, 40), cmap='Blues')
    colors = ['gray', 'red', 'orange', 'lime', 'purple', 'yellow', 'magenta']
    for i, (k, t) in enumerate(tt.items()):
        t = model.transform(t)
        loss = pd.read_csv(f"results/alignment/loss_{k}.csv").iloc[seed].final_loss
        ax.plot(
            t[:, 0], t[:, 1], lw=2, label=f"{k} (loss={loss:.3f})", color=colors[i], alpha=0.5,
            path_effects=[pe.Stroke(linewidth=3, foreground='white', alpha=0.5), pe.Normal()])
        ax.scatter(t[0, 0], t[0, 1], color='white', zorder=3)
        ax.scatter(t[-1, 0], t[-1, 1], color='black')
    fig.legend(loc='lower center', mode='expand', ncol=4)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Contours Plot')
    fig.savefig(save_path)
    plt.close()
