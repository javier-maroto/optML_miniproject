import logging
import tensorflow as tf
import numpy as np
from oml.angles import euler6tomarix4d, d_q, euler2quaternion
from tensorflow.keras.optimizers import Adam
import time
import seaborn as sns
import matplotlib.pyplot as plt


def update_quaternion(m, a_R, q_old, transposed=True):
    """Update old quaternion with learned rotation

    m: np.array
        Array of shape (4,) containing diagonal values for rotation matrix. It is used for flipping the rotation.
        When diagonal value is -1, it is flipped. Otherwise, it is identity.
    a_R: np.array
        Array of shape (6,) containing the rotation angles used to align the old quaternion to the true one.
    q_old: np.ndarray
        Array of shape (N, 4). Quaternions that we will align.
    """
    # 4D matrix rotation
    R = euler6tomarix4d(a_R)
    I = tf.linalg.diag(tf.convert_to_tensor(m, dtype=tf.float64))
    q_new = tf.transpose(R @ I @ tf.transpose(q_old))

    return q_new


def loss_alignment(m, a_R, q_predicted, q_true, transposed):
    """Loss for optimization"""
    # 4D matrix rotation
    R = euler6tomarix4d(a_R)
    I = tf.linalg.diag(tf.convert_to_tensor(m, dtype=tf.float64))
    q_predicted_rotated = tf.transpose(R @ I @ tf.transpose(q_predicted))

    return tf.reduce_mean(d_q(q_true, q_predicted_rotated))


def gradient_alignment(m, a_R, q_predicted, q_true, transposed):
    """Gradion for optimization"""
    with tf.GradientTape() as tape:
        loss_value = loss_alignment(m, a_R, q_predicted, q_true, transposed)
        gradient = tape.gradient(loss_value, a_R)

    return loss_value, gradient


def training_angle_alignment(
        m, steps, batch_size, projection_idx, learning_rate,
        angles_true, angles_predicted, transposed=True, seed=0):
    """Optimization"""
    optimizer = Adam(learning_rate=learning_rate)

    time_start = time.time()

    losses = np.empty(steps)
    angles_predicted = tf.convert_to_tensor(angles_predicted)

    trajectory = np.zeros([steps + 1, 6])
    euler = tf.random.uniform([6], 0, 2*np.pi, dtype=tf.float64, seed=seed)
    a_R = [tf.Variable(euler)]
    trajectory[0, :] = a_R[0].numpy()

    q_predicted = euler2quaternion(angles_predicted)
    q_true = euler2quaternion(angles_true)

    for step in range(1, steps+1):

        # Sample some pairs.
        idx = list(np.random.choice(projection_idx, size=batch_size))

        # Compute distances between projections
        qt = [q_true[i]      for i in idx]
        qp = [q_predicted[i] for i in idx]

        # Optimize by gradient descent.
        losses[step-1], gradients = gradient_alignment(
            m, a_R, qp, qt, transposed)
        optimizer.apply_gradients(zip(gradients, a_R))
        trajectory[step, :] = a_R[0].numpy()

        update_lr = 300
        if ((step > update_lr) and (step % update_lr == 0) and
                (losses[step-1]-losses[step-1-update_lr+100] < 0.1)):
            learning_rate *= 0.1

        # Visualize progress periodically
        if step % 10 == 0:
            plt.close()
            sns.set(style="white", color_codes=True)
            sns.set(style="whitegrid")

            fig, axs = plt.subplots(1, 3, figsize=(24, 7))

            # Distance count subplot (batches)
            qpr = update_quaternion(m, a_R, qp, transposed=transposed)
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
                f"LR={learning_rate:.2e}")

            # Distance count subplot (full)
            q_predicted_rot = update_quaternion(
                m, a_R, q_predicted, transposed=transposed)
            d2 = d_q(q_predicted_rot, q_true)
            axs[2].set_xlim(0, np.pi)
            # axs[2].set_ylim(0, len(angles_true))
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
            fig.savefig(f'results/alignment/plots/training_{seed}.png')

        # Periodically report progress.
        if ((step % (steps//10)) == 0) or (step == steps):
            time_elapsed = time.time() - time_start
            logging.info(
                f'step {step}/{steps} ({time_elapsed:.0f}s): '
                f'loss = {np.mean(losses[step-steps//10:step-1]):.2e}')

        if step >= 101 and np.mean(losses[step-101:step-1]) < 1e-3:
            break
    np.save(
        f"results/alignment/trajectories/{seed}.npy", trajectory[:step + 1])
    return m, a_R, np.mean(losses[-1-steps//10:-1])
