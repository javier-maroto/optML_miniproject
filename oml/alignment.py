import logging
import tensorflow as tf
import numpy as np
from .angles import d_q, euler2quaternion
import time
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from IPython import display as IPyDisplay


def euler6tomarix4d(a_R):
    """ Convert 6D vector containing angles to 4D rotation matrix

    Parameters
    ----------
    a_R: tf.tensor/np.ndarray
        Vector of shape (6,)

    Returns
    -------
    R: tf.tensor/np.ndarray
        4x4 Rotation matrix corresponding to these 6 angles of rotations
    """
    xy, xz, xw, yz, yw, zw = tf.unstack(a_R, axis=-1)

    cxy = tf.cos(xy)
    cxz = tf.cos(xz)
    cxw = tf.cos(xw)
    cyz = tf.cos(yz)
    cyw = tf.cos(yw)
    czw = tf.cos(zw)

    sxy = tf.sin(xy)
    sxz = tf.sin(xz)
    sxw = tf.sin(xw)
    syz = tf.sin(yz)
    syw = tf.sin(yw)
    szw = tf.sin(zw)

    # Note: wasn't able to create it as simple as np.ndarrays...
    Rxy = [[  cxy,  -sxy, [0.0], [0.0]],
           [  sxy,   cxy, [0.0], [0.0]],
           [[0.0], [0.0], [1.0], [0.0]],
           [[0.0], [0.0], [0.0], [1.0]]]
    Rxy = tf.reshape(tf.convert_to_tensor(Rxy, dtype=tf.float64), (4, 4))

    Rxz = [[  cxz, [0.0],  -sxz, [0.0]],
           [[0.0], [1.0], [0.0], [0.0]],
           [  sxz, [0.0],   cxz, [0.0]],
           [[0.0], [0.0], [0.0], [1.0]]]
    Rxz = tf.reshape(tf.convert_to_tensor(Rxz, dtype=tf.float64), (4, 4))

    Rxw = [[  cxw, [0.0], [0.0],  -sxw],
           [[0.0], [1.0], [0.0], [0.0]],
           [[0.0], [0.0], [1.0], [0.0]],
           [  sxw, [0.0], [0.0],  cxw]]
    Rxw = tf.reshape(tf.convert_to_tensor(Rxw, dtype=tf.float64), (4, 4))

    Ryz = [[[1.0], [0.0], [0.0], [0.0]],
           [[0.0],   cyz,  -syz, [0.0]],
           [[0.0],   syz,   cyz, [0.0]],
           [[0.0], [0.0], [0.0], [1.0]]]
    Ryz = tf.reshape(tf.convert_to_tensor(Ryz, dtype=tf.float64), (4, 4))

    Ryw = [[[1.0], [0.0], [0.0], [0.0]],
           [[0.0],   cyw, [0.0],  -syw],
           [[0.0], [0.0], [1.0], [0.0]],
           [[0.0],   syw, [0.0],  cyw]]
    Ryw = tf.reshape(tf.convert_to_tensor(Ryw, dtype=tf.float64), (4, 4))

    Rzw = [[[1.0], [0.0], [0.0], [0.0]],
           [[0.0], [1.0], [0.0], [0.0]],
           [[0.0], [0.0],   czw,  -szw],
           [[0.0], [0.0],   szw,  czw]]
    Rzw = tf.reshape(tf.convert_to_tensor(Rzw, dtype=tf.float64), (4, 4))

    R = Rxy @ Rxz @ Rxw @ Ryz @ Ryw @ Rzw

    # check if matrix is orthogonal: R^T @ R - I < 1e-5
    assert tf.reduce_all(tf.less_equal(
        tf.abs(tf.subtract(
            tf.transpose(R) @ R, tf.eye(4, 4, dtype=tf.float64))),
        1e-5)), "Matrix R (4x4) should be orthogonal!"

    return R

def update_quaternion(m, a_R, q_old):
    """Update old quaternion with learned rotation

    Parameters
    ----------
    m:  list of 4 floats (-1 or 1)
        List of length 4 containing diagonal values for rotation matrix. It is used for flipping the rotation.
        When diagonal value is -1, it is flipped. Otherwise, it is identity.
    a_R: np.array
        Array of shape (6,) containing the rotation angles used to align the old quaternion to the true one.
    q_old: np.ndarray
        Array of shape (N, 4). Quaternions that we will align.

    Returns
    -------
    q_new: np.ndarray
        Array of shape (N, 4). Rotated quaternion
    """
    # 4D matrix rotation
    R = euler6tomarix4d(a_R)
    I = tf.linalg.diag(tf.convert_to_tensor(m, dtype=tf.float64))
    q_new = tf.transpose(R @ I @ tf.transpose(q_old))

    return q_new


def loss_alignment(m, a_R, q_predicted, q_true):
    """Loss for optimization
    
    Parameters
    ----------
    m: list of 4 floats (-1 or 1)
        List of length 4 containing diagonal values for rotation matrix. It is used for flipping the rotation.
        When diagonal value is -1, it is flipped. Otherwise, it is identity.
    a_R: np.array
        Array of shape (6,) containing the rotation angles used to align the old quaternion to the true one.
    q_predicted: np.ndarray
        Array of shape (N, 4). Predicted/estimated quaternion.
    q_true: np.ndarray
        Array of shape (N, 4). Ground-truth quaternion.

    Returns
    -------
        Loss function specified as a mean of distances between true and predicted quaternion.
    """
    # 4D matrix rotation
    R = euler6tomarix4d(a_R)
    I = tf.linalg.diag(tf.convert_to_tensor(m, dtype=tf.float64))
    q_predicted_rotated = tf.transpose(R @ I @ tf.transpose(q_predicted))

    return tf.reduce_mean(d_q(q_true, q_predicted_rotated))


def gradient_alignment(m, a_R, q_predicted, q_true):
    """Gradion for optimization
     
    Parameters
    ----------
    m: list of 4 floats (-1 or 1)
        List of length 4 containing diagonal values for rotation matrix. It is used for flipping the rotation.
        When diagonal value is -1, it is flipped. Otherwise, it is identity.
    a_R: np.array
        Array of shape (6,) containing the rotation angles used to align the old quaternion to the true one.
    q_predicted: np.ndarray
        Array of shape (N, 4). Predicted/estimated quaternion.
    q_true: np.ndarray
        Array of shape (N, 4). Ground-truth quaternion.

    Returns
    -------
    loss_value: tf.tensor
        Loss value for the optimization
    gradient: tf.tensor
        Gradient value
    """
    with tf.GradientTape() as tape:
        loss_value = loss_alignment(m, a_R, q_predicted, q_true)
        gradient = tape.gradient(loss_value, a_R)

    return loss_value, gradient


def training_angle_alignment(
        m, steps, batch_size, optimizer,
        angles_true, angles_predicted, projection_idx=None, seed=0, plot_interactive=False):
    """Optimization
    
    
    """
    collect_data = []
    
    opt_name = optimizer.__class__.__name__
    lr = optimizer.lr.numpy()
    Path(f"results/alignment/plots/{opt_name}").mkdir(parents=True, exist_ok=True)
    Path(f"results/alignment/trajectories/{opt_name}").mkdir(parents=True, exist_ok=True)

    time_start = time.time()

    losses = np.empty(steps)
    angles_predicted = tf.convert_to_tensor(angles_predicted)

    trajectory = np.zeros([steps + 1, 6])
    tf.random.set_seed(seed)
    euler = tf.random.uniform([6], 0, 2*np.pi, dtype=tf.float64, seed=seed)
    a_R = [tf.Variable(euler)]
    trajectory[0, :] = a_R[0].numpy()

    q_predicted = euler2quaternion(angles_predicted)
    q_true = euler2quaternion(angles_true)

    projection_idx = projection_idx or range(len(angles_true))

    for step in range(1, steps+1):

        # Sample some pairs.
        idx = list(np.random.choice(projection_idx, size=batch_size))

        # Compute distances between projections
        qt = [q_true[i]      for i in idx]
        qp = [q_predicted[i] for i in idx]

        # Optimize by gradient descent.
        losses[step-1], gradients = gradient_alignment(m, a_R, qp, qt)
        optimizer.apply_gradients(zip(gradients, a_R))
        trajectory[step, :] = a_R[0].numpy()

        # Visualize progress periodically
        if step % 10 == 0:
            qu = update_quaternion(m, a_R, q_predicted)
            collect_data.append(qu.numpy())

            plt.close()
            sns.set(style="white", color_codes=True)
            sns.set(style="whitegrid")

            fig, axs = plt.subplots(1, 3, figsize=(24, 7))

            # Distance count subplot (batches)
            qpr = update_quaternion(m, a_R, qp)
            d1 = d_q(qpr, qt)
            axs[0].set_xlim(0, np.pi)
            axs[0].set_title(
                f"BATCHES (size={len(qp)}): [{step}/{steps}] "
                f"Distances between true and predicted angles \n"
                f"MEAN={np.mean(d1):.2e} STD={np.std(d1):.2e}")
            s = sns.distplot(
                d1, kde=False, bins=100, ax=axs[0],
                axlabel="Distance [rad]", color="r")
            max_count = int(max([h.get_height() for h in s.patches]))
            axs[0].plot(
                [np.mean(d1)]*max_count, np.arange(0, max_count), c="r", lw=4)

            # Optimization loss subplot
            loss = np.mean(losses[step-10:step])
            axs[1].plot(
                np.linspace(0, time.time()-time_start, step),
                losses[:step], marker="o", lw=1, markersize=3)
            axs[1].set_xlabel('time [s]')
            axs[1].set_ylabel('loss')
            axs[1].set_title(
                f"Angle alignment optimization \nLOSS={loss:.2e} "
                f"LR={lr:.2e}")

            # Distance count subplot (full)
            q_predicted_rot = update_quaternion(m, a_R, q_predicted)
            d2 = d_q(q_predicted_rot, q_true)
            axs[2].set_xlim(0, np.pi)
            axs[2].set_title(
                f"FULL: [{step}/{steps}] Distances between true and "
                f"predicted angles\n"
                f"MEAN={np.mean(d2):.2e} ({np.degrees(np.mean(d2)):.2e} deg) "
                f"STD={np.std(d2):.2e}\nMEDIAN={np.median(d2):.2e} "
                f"({np.degrees(np.median(d2)):.2e} deg)")
            s = sns.distplot(
                d2, kde=False, bins=100, ax=axs[2],
                axlabel="Distance [rad]", color="r")
            max_count = int(max([h.get_height() for h in s.patches]))
            axs[2].plot(
                [np.mean(d2)]*max_count, np.arange(0, max_count), c="r", lw=4)
            fig.savefig(f'results/alignment/plots/{opt_name}/training_{seed}.png')

            if plot_interactive:
                IPyDisplay.clear_output(wait=True)
                IPyDisplay.display(plt.gcf())
                plt.close();
                time.sleep(.1)

        # Periodically report progress.
        if ((step % (steps//10)) == 0) or (step == steps):
            time_elapsed = time.time() - time_start
            logging.debug(
                f'step {step}/{steps} ({time_elapsed:.0f}s): '
                f'loss = {np.mean(losses[step-steps//10:step-1]):.2e}')

        # if loss hasn't change in the last 100 steps, break from the loop
        if step >= 101 and np.mean(losses[step-101:step-1]) < 1e-3:
            break;
    np.save(
        f"results/alignment/trajectories/{opt_name}/{seed}.npy", trajectory[:step + 1])
    
    return a_R, np.mean(losses[-1-steps//10:-1]), np.array(collect_data), trajectory

def angles_transpose(angles):
    """Transpose the angles
    
    Parameters
    ----------
    angles: np.ndarray
        Angles we want to transpose
    
    Returns
    -------
    angles: np.ndarray
        Transposed angles
    """
    cols = [2, 1, 0]
    idx = np.empty_like(cols)
    idx[cols] = np.arange(len(cols))
    angles[:] = -angles[:, idx]
    return angles