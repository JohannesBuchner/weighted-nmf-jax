import numpy as np
from wNMFx import wNMF
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose


def test_random(plot=False):
    ## An example on simulated data
    n = 101
    features = 100
    components = 4
    n_run = 10
    max_iter = 4000
    np.random.seed(123)
    
    shapes_true = np.array([0.5 + 0.5 * np.sin(np.arange(features) / 10 / 2**i + np.random.uniform(0, np.pi)) for i in range(components)])
    shapes_true[0] = 1
    shapes_true[1] = np.exp(-np.arange(features) / 400)

    if plot:
        plt.plot(shapes_true.T);
        plt.savefig('test_normal_input.pdf')
        plt.close()

    X = np.abs(np.random.normal(10 * (10**np.random.uniform(-1, 1, size=(n, components)) @ shapes_true), 1))
    assert X.shape == (n, features)
    W = np.ones_like(X)
    if plot:
        plt.plot(X[:10].T, ls=' ', marker='o');
        plt.savefig('test_normal_data.pdf')
        plt.close()
        fig_components, ax_components = plt.subplots()
        fig_loss, ax_loss = plt.subplots()

    for i, (init, ls) in enumerate([('random', '-'), ('sample', '--'), ('PCA', ':'), ('logAR1', ':')]):
        model = wNMF(
            n_components=components, beta_loss='frobenius', 
            max_iter=max_iter, track_error=True, verbose=1, init=init, 
            n_run=n_run)
        fit = model.fit(X=X, W=W)
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

        model.tol = 0
        V = model.transform(X=X, W=W)
        V2 = model.transform(X=X * 100, W=W / 100**2)
        assert_allclose(V * 100, V2, rtol=0.01)

    if plot:
        fig_components.savefig('test_normal.pdf')
        plt.close(fig_components)
        ax_loss.set_yscale('log')
        ax_loss.legend()
        fig_loss.savefig('test_normal_loss.pdf')
        plt.close(fig_loss)


def test_poisson(plot=False):
    ## An example on simulated data
    n = 101
    features = 1000
    components = 4
    n_run = 10
    max_iter = 2000
    np.random.seed(1)

    shapes_true = np.array([0.5 + 0.5 * np.sin(np.arange(features) / 10 / 2**i + np.random.uniform(0, np.pi)) for i in range(components)])
    shapes_true[0] = 1
    shapes_true[1] = np.exp(-np.arange(features) / 400)
    shapes_true /= shapes_true.max(axis=0, keepdims=True)

    if plot:
        for k, component in enumerate(shapes_true):
            plt.plot(component, label=f'component {k}')
        plt.legend()
        plt.savefig('test_poisson_input.pdf')
        plt.close()

    ## An example on simulated data
    X = 1. * np.random.poisson(100 * (10**np.random.uniform(-4, 2, size=(n, components))) @ shapes_true)
    W = np.ones_like(X)
    assert X.shape == (n, features)
    if plot:
        plt.plot(X[:10].T, ls=' ', marker='o');
        plt.savefig('test_poisson_data.pdf')
        plt.close()
        fig_components, ax_components = plt.subplots()
        fig_loss, ax_loss = plt.subplots()

    for i, (init, ls) in enumerate([('random', '-'), ('sample', '--'), ('PCA', ':'), ('logAR1', ':')]):
        model = wNMF(
            n_components=components, beta_loss='kullback-leibler', 
            max_iter=max_iter, track_error=True, verbose=1, init=init, 
            n_run=n_run)
        fit = model.fit(X=X, W=W)
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
        fig_components.savefig('test_poisson.pdf')
        plt.close(fig_components)
        ax_loss.set_yscale('log')
        ax_loss.legend()
        fig_loss.savefig('test_poisson_loss.pdf')
        plt.close(fig_loss)


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'poisson':
        test_poisson(plot=True)
    elif sys.argv[1] == 'normal':
        test_random(plot=True)
