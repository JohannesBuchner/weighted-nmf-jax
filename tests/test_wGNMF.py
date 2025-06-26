import numpy as np
from wNMFx import wNMF, wGNMF
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose


def test_random(plot=False):
    ## An example on simulated data
    components = 2
    n_run = 1
    max_iter = 400
    np.random.seed(123)
    mulmask = np.array([False, True])
    
    x = np.arange(10, 15, 0.25)
    features = len(x)
    true_shape = (x/x[0])**-1
    absorber = np.sin(x)**2
    
    X = []
    for i in np.logspace(-3, -1, 21):
        ampl = np.random.uniform(98, 100)
        mod_shape = true_shape * absorber**i
        if plot:
            plt.plot(mod_shape, lw=0.2)
        X.append(ampl * mod_shape)
    n = len(X)
    if plot:
        plt.savefig('test_GNMF_input.pdf')
        plt.close()
    
    X = np.random.normal(X, 1)
    assert X.shape == (n, features)
    W = np.ones_like(X)
    
    if plot:
        plt.plot(X[:10].T, ls=' ', marker='o');
        plt.savefig('test_GNMF_data.pdf')
        plt.close()
        fig_components, ax_components = plt.subplots()
        fig_loss, ax_loss = plt.subplots()

    init = 'random'
    ls = '-'
    model = wGNMF(
        n_components=components, beta_loss='frobenius', 
        max_iter=max_iter, track_error=True, verbose=2, init=init, 
        n_run=n_run, tol=1e-5)
    fit = model.fit(X=X, W=W, mulmask=mulmask)
    print(fit.V.shape)
    print(fit.U.shape)
    assert fit.V.shape == (components, n)
    assert fit.U.shape == (features, components)
    assert np.shape(fit.err) == ()
    assert np.shape(fit.err) == ()
    assert len(fit.error_tracker) == n_run
    assert len(fit.error_tracker[0]) == max_iter

    if plot:
        ax_components.plot(i + fit.U, ls=ls)
        color = None
        for i, err_tracked in enumerate(fit.error_tracker):
            l, = ax_loss.plot(err_tracked, color=color,
                label=f'run {i} init {init}' if color is None else None, ls=ls)
            color = l.get_color()

    if plot:
        fig_components.savefig('test_GNMF.pdf')
        plt.close(fig_components)
        ax_loss.set_yscale('log')
        ax_loss.legend()
        fig_loss.savefig('test_GNMF_loss.pdf')
        plt.close(fig_loss)


if __name__ == '__main__':
    import sys
    test_random(plot=True)
