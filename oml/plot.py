from oml.angles import create_unique_angle
import numpy as np
import matplotlib.pyplot as plt


def plot_contour(t, good_point, w_good=10):
    _, fun, diff1, diff2 = create_unique_angle(t[-1], ret_full=True)
    t = np.array([[fun[i](tt[i] + diff1[i]) + diff2[i] for i in range(6)] for tt in t])
    t = np.concatenate([t] + [good_point] * w_good)

    model = PCA(n_components=2)
    t = model.fit_transform(t)
    ls_dim = lambda dim: np.linspace(t[:,dim].min(), t[:,dim].max(), RES[dim])
    X, Y = np.meshgrid(ls_dim(0), ls_dim(1))
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            angle = model.inverse_transform((X[i,j], Y[i,j]))
            angle = [tf.Variable(angle)]
            Z[i,j] = loss_alignment([1., 1., 1., 1.], angle, q_pred, q_true)

    fig,ax=plt.subplots(1,1, figsize=(15,15))
    cp = ax.contourf(X, Y, Z, levels=40)
    ax.plot(t[:-w_good,0], t[:-w_good,1], color='orange')
    ax.scatter(t[-w_good-1,0], t[-w_good-1,1], color='red', zorder=3)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Contours Plot')
    return fig