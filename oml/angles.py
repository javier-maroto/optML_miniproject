import math

import tensorflow as tf
import numpy as np
from tensorflow_graphics.util import asserts, safe_ops, shape
from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.math import vector


def euler2quaternion(angles, transposed=True):
    """Convert Euler angles to quaternion

    Parameters
    ----------
    angles: np.ndarray
        Array of shape (N, 3), where 3 are the 3 anles of rotation
        around Z-Y-Z axes respectivelly.

    Returns
    -------
    quaternion: np.ndarray
        Array of shape (N, 4), where 4 is the 1 real part of quaternion
        and 3 imaginary parts of quaternion.
    """
    angles = tf.convert_to_tensor(value=angles)

    shape.check_static(
        tensor=angles, tensor_name="angles", has_dim_equals=(-1, 3))

    theta_z1, theta_y, theta_z0 = tf.unstack(angles, axis=-1)

    # create rotation matrix
    c1 = tf.cos(theta_z1)
    c2 = tf.cos(theta_y)
    c3 = tf.cos(theta_z0)

    s1 = tf.sin(theta_z1)
    s2 = tf.sin(theta_y)
    s3 = tf.sin(theta_z0)

    if not transposed:
        r00 = c1*c2*c3-s1*s3
        r10 = c1*s3+c2*c3*s1
        r20 = -c3*s2
        r01 = -c3*s1-c1*c2*s3
        r11 = c1*c3-c2*s1*s3
        r21 = s2*s3
        r02 = c1*s2
        r12 = s1*s2
        r22 = c2

    else:
        # PROJECTIONS CODE
        r00 = c1*c2*c3-s1*s3
        r01 = c1*s3+c2*c3*s1
        r02 = -c3*s2
        r10 = -c3*s1-c1*c2*s3
        r11 = c1*c3-c2*s1*s3
        r12 = s2*s3
        r20 = c1*s2
        r21 = s1*s2
        r22 = c2

    w2 = 1/4*(1 + r00 + r11 + r22)
    w2_is_pos = tf.greater(w2, 0)

    x2 = -1/2*(r11+r22)
    x2_is_pos = tf.greater(x2, 0)

    y2 = 1/2*(1-r22)
    y2_is_pos = tf.greater(y2, 0)

    w = tf.compat.v1.where(w2_is_pos, tf.sqrt(w2), tf.zeros_like(w2))
    x = tf.compat.v1.where(
        w2_is_pos, 1/(4*w)*(r21-r12),
        tf.compat.v1.where(x2_is_pos, tf.sqrt(x2), tf.zeros_like(x2)))
    y = tf.compat.v1.where(
        w2_is_pos, 1/(4*w)*(r02-r20),
        tf.compat.v1.where(
            x2_is_pos, r01/(2*x),
            tf.compat.v1.where(y2_is_pos, tf.sqrt(y2), tf.zeros_like(y2))))

    z = tf.compat.v1.where(
        w2_is_pos, 1/(4*w)*(r10-r01),
        tf.compat.v1.where(
            x2_is_pos, r02/(2*x),
            tf.compat.v1.where(y2_is_pos, r12/(2*y), tf.ones_like(y2))))

    return tf.stack((x, y, z, w), axis=-1)


def quaternion2euler(quaternions, transposed=True):
    """ Convert quaternion to Euler angles

    Parameters
    ----------
    quaternions: np.ndarray
        The array of shape (N, 4), where 4 is the 1 real part of quaternion
        and 3 imaginary parts of quaternion.

    Returns
    -------
    angles: np.ndarray
        The array of shape (N, 3), where 3 are the 3 anles of rotation
        around Z-Y-Z axes respectivelly.
    """
    def general_case(r02, r12, r20, r21, r22, eps_addition):
        """Handles the general case."""
        theta_y = tf.acos(r22)
        # sign_sin_theta_y = safe_ops.nonzero_sign(tf.sin(theta_y))

        r02 = safe_ops.nonzero_sign(r02) * eps_addition + r02
        r22 = safe_ops.nonzero_sign(r22) * eps_addition + r22

        theta_z0 = tf.atan2(r12, r02)
        theta_z1 = tf.atan2(r21, -r20)
        return tf.stack((theta_z0, theta_y, theta_z1), axis=-1)

    def gimbal_lock(r22, r11, r10, eps_addition):
        """Handles Gimbal locks.
        It is gimbal when r22 is -1 or 1"""
        sign_r22 = safe_ops.nonzero_sign(r22)
        r11 = safe_ops.nonzero_sign(r11) * eps_addition + r11

        theta_z0 = tf.atan2(sign_r22 * r10, r11)

        theta_y = (
            tf.constant(math.pi/2.0, dtype=r20.dtype)
            - sign_r22 * tf.constant(math.pi/2.0, dtype=r20.dtype))
        theta_z1 = tf.zeros_like(theta_z0)
        angles = tf.stack((theta_z0, theta_y, theta_z1), axis=-1)
        return angles

    with tf.compat.v1.name_scope(None, "euler_from_quaternion", [quaternions]):
        quaternions = tf.convert_to_tensor(value=quaternions)

        shape.check_static(
            tensor=quaternions,
            tensor_name="quaternions",
            has_dim_equals=(-1, 4))

        x, y, z, w = tf.unstack(quaternions, axis=-1)
        tx = safe_ops.safe_shrink(2.0 * x, -2.0, 2.0, True)
        ty = safe_ops.safe_shrink(2.0 * y, -2.0, 2.0, True)
        tz = safe_ops.safe_shrink(2.0 * z, -2.0, 2.0, True)
        twx = tx * w
        twy = ty * w
        twz = tz * w
        txx = tx * x
        txy = ty * x
        txz = tz * x
        tyy = ty * y
        tyz = tz * y
        tzz = tz * z

        # The following is clipped due to numerical instabilities
        # that can take some enties outside the [-1;1] range.

        if not transposed:
            # r00 = safe_ops.safe_shrink(1.0 - (tyy + tzz), -1.0, 1.0, True)
            # r01 = safe_ops.safe_shrink(txy - twz, -1.0, 1.0, True)
            r02 = safe_ops.safe_shrink(txz + twy, -1.0, 1.0, True)

            r10 = safe_ops.safe_shrink(txy + twz, -1.0, 1.0, True)
            r11 = safe_ops.safe_shrink(1.0 - (txx + tzz), -1.0, 1.0, True)
            r12 = safe_ops.safe_shrink(tyz - twx, -1.0, 1.0, True)

            r20 = safe_ops.safe_shrink(txz - twy, -1.0, 1.0, True)
            r21 = safe_ops.safe_shrink(tyz + twx, -1.0, 1.0, True)
            r22 = safe_ops.safe_shrink(1.0 - (txx + tyy), -1.0, 1.0, True)

        else:
            # r00 = safe_ops.safe_shrink(1.0 - (tyy + tzz), -1.0, 1.0, True)
            # r01 = safe_ops.safe_shrink(txy + twz, -1.0, 1.0, True)
            r02 = safe_ops.safe_shrink(txz - twy, -1.0, 1.0, True)

            r10 = safe_ops.safe_shrink(txy - twz, -1.0, 1.0, True)
            r11 = safe_ops.safe_shrink(1.0 - (txx + tzz), -1.0, 1.0, True)
            r12 = safe_ops.safe_shrink(tyz + twx, -1.0, 1.0, True)

            r20 = safe_ops.safe_shrink(txz + twy, -1.0, 1.0, True)
            r21 = safe_ops.safe_shrink(tyz - twx, -1.0, 1.0, True)
            r22 = safe_ops.safe_shrink(1.0 - (txx + tyy), -1.0, 1.0, True)

        eps_addition = asserts.select_eps_for_addition(quaternions.dtype)
        general_solution = general_case(r02, r12, r20, r21, r22, eps_addition)
        gimbal_solution = gimbal_lock(r22, r11, r10, eps_addition)

        # The general solution is unstable close to the Gimbal lock,
        # and the gimbal solution is not toooff in these cases.
        # Check if r22 is 1 or -1
        is_gimbal = tf.less(tf.abs(tf.abs(r22) - 1.0), 1.0e-6)
        gimbal_mask = tf.stack((is_gimbal, is_gimbal, is_gimbal), axis=-1)

        return tf.compat.v1.where(
            gimbal_mask, gimbal_solution, general_solution)


def d_q(q1, q2):
    """Distance between 2 quaternions

    The quaternion distance takes values between [0, pi]

    Parameters
    ----------
    q1: tf.tensor/np.ndarray
        1st quaternion
    q2: tf.tensor/np.ndarray
        2nd quaternion

    Returns
    -------
    : distnace between these 2 quaternions

    """
    q1 = tf.cast(tf.convert_to_tensor(value=q1), dtype=tf.float64)
    q2 = tf.cast(tf.convert_to_tensor(value=q2), dtype=tf.float64)

    shape.check_static(
        tensor=q1, tensor_name="quaternion1", has_dim_equals=(-1, 4))
    shape.check_static(
        tensor=q2, tensor_name="quaternion2", has_dim_equals=(-1, 4))

    q1 = quaternion.normalize(q1)
    q2 = quaternion.normalize(q2)

    dot_product = vector.dot(q1, q2, keepdims=False)

    # Ensure dot product is in range [-1. 1].
    eps_dot_prod = 1.8 * asserts.select_eps_for_addition(dot_product.dtype)
    dot_product = safe_ops.safe_shrink(
        dot_product, -1, 1, open_bounds=False, eps=eps_dot_prod)

    return 2.0 * tf.acos(tf.abs(dot_product))


def distance_difference(angles_predicted, angles_true):
    """Average quaternion distance between true and predicted quaternions"""
    q_predicted = euler2quaternion(angles_predicted)
    q_true = euler2quaternion(angles_true)
    qd = np.mean(d_q(q_predicted, q_true).numpy())
    print(
        "Mean `quaternion` distance between true and predicted values:"
        f"{qd:.3f} rad ({np.degrees(qd):.3f} degrees)")

    return qd


def euler6tomarix4d(a_R):
    """ Convert 6D vector containing angles to 4D rotation matrix

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


def rotations_equal(R1, R2):
    R1 = list(map(lambda x: x % (2*np.pi), R1))
    R2 = list(map(lambda x: x % (2*np.pi), R2))

    r = (
        lambda i: R1[i]+R2[i] -
        2*(np.round(R1[i]-R2[i], 2) % np.round(np.pi, 2) == 0) * R2[i])

    rd1 = np.round(np.round(r(0)+r(5), 2) % np.round(2*np.pi, 2), 2)
    rd2 = np.round(np.round(r(1)+r(4), 2) % np.round(2*np.pi, 2), 2)
    rd3 = np.round(np.round(r(2)+r(3), 2) % np.round(2*np.pi, 2), 2)

    r = (rd1+rd2+rd3) % (3*np.round(np.pi, 2))

    return round(r, 1) == 0


def create_unique_angle(point, ret_full=False):
    """Constructs the equivalent 6d vector with all positive components
    such that their sum is minimum"""
    add_pi = lambda x: x + np.pi
    opp = lambda x: -x
    opp_pi = lambda x: np.pi - x
    idd = lambda x: x
    LIST_FUN = [idd, opp, add_pi, opp_pi]
    LIST_EQUIV = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 3, 2],
        [0, 0, 2, 0, 3, 2],
        [0, 0, 2, 2, 0, 0],
        [0, 2, 1, 1, 2, 0],
        [0, 2, 1, 3, 1, 2],
        [0, 2, 3, 1, 1, 2],
        [0, 2, 3, 3, 2, 0],
        [2, 1, 1, 1, 1, 2],
        [2, 1, 1, 3, 2, 0],
        [2, 1, 3, 1, 2, 0],
        [2, 1, 3, 3, 1, 2],
        [2, 3, 0, 0, 3, 2],
        [2, 3, 0, 2, 0, 0],
        [2, 3, 2, 0, 0, 0],
        [2, 3, 2, 2, 3, 2]
    ]
    p = [x % (2 * np.pi) for x in point]
    best_sum = np.inf
    best_point = None
    best_eq = None
    diff_1 = np.array(p) - np.array(point)
    diff_2 = np.zeros(6)
    for eq in LIST_EQUIV:
        x = np.zeros_like(p)
        _diff_2 = np.zeros(6)
        for i in range(6):
            x[i] = LIST_FUN[eq[i]](p[i])
            if x[i] < 0:
                x[i] += 2 * np.pi
                _diff_2[i] += 2 * np.pi
            elif x[i] > (2 * np.pi):
                x[i] -= 2 * np.pi
                _diff_2[i] -= 2 * np.pi
        if sum(x) == best_sum:
            # Ties are solved by smaller first components
            for i in range(6):
                if x[i] == best_point[i]:
                    continue
                elif x[i] < best_point[i]:
                    best_point = x
                    best_eq = eq
                    diff_2 = _diff_2.copy()
                    break
                else:
                    break
        elif sum(x) < best_sum:
            best_point = x
            best_sum = sum(x)
            best_eq = eq
            diff_2 = _diff_2.copy()
    if ret_full:
        best_eq = [LIST_FUN[best_eq[i]] for i in range(6)]
        return best_point, best_eq, diff_1, diff_2
    return best_point