import numpy as np
from wNMFx import wNMF
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

from wNMFx.wNMF import calculate_reconstruction_error_frobenius, update_uv_batch_frobenius, iterate_UV, decorrelate_NMF_greedy

def test_evolution(plot=False):
    ## An example on simulated data
    n = 30001
    features = 40
    components = 4
    max_iter = 100
    np.random.seed(125)
    
    shapes_true = np.array([0.5 + 0.5 * np.sin(np.arange(features) / 10 / 2**i + np.random.uniform(0, np.pi)) for i in range(components)])
    shapes_true[0] = 1
    shapes_true[1] = np.exp(-np.arange(features) / 400)

    if plot:
        plt.plot(shapes_true.T);
        plt.savefig('test_normal_input.pdf')
        plt.close()

    X = np.abs(np.random.normal(10 * (10**np.random.uniform(-1, 1, size=(n, components)) @ shapes_true), 1)).T
    assert X.shape == (features, n)
    W = np.ones_like(X)
    if plot:
        plt.plot(X[:,:10], ls=' ', marker='o');
        plt.savefig('test_normal_data.pdf')
        plt.close()

    rng = np.random
    scale = np.mean(X) / components
    # generate entries of U/V using randn, scale by est
    U = rng.uniform(size=(features, components))
    V = np.abs(scale * rng.randn(components, n))

    calculate_reconstruction_error_func = calculate_reconstruction_error_frobenius
    update_uv_batch_func = update_uv_batch_frobenius
    nexamples = 10
    fig_rec, axs_rec = plt.subplots(nexamples, 1, figsize=(10, 14), sharex=True, gridspec_kw=dict(hspace=0))
    fig_comp, axs_comp = plt.subplots(components, 1, figsize=(10, 8), sharex=True, gridspec_kw=dict(hspace=0))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, max_iter))
    for j in range(components):
        axs_comp[j].plot(U[:,j], color='k', alpha=0.5)
        axs_comp[j].set_ylabel(f'Component {j}')
        #axs_comp[j].set_yscale('log')
        axs_comp[j].set_ylim(0, 1)
    for j in range(nexamples):
        axs_rec[j].plot(X[:,j], ' o', color='k')
        axs_rec[j].set_ylabel(f'Sample {j}')
        axs_rec[j].set_ylim(axs_rec[j].get_ylim())
        #axs_rec[j].set_ylim(X[:,j].min(), X[:,j].max())
    for (U, V, i, err, err_stored), color in zip(iterate_UV(
        X, U, V, W, R=None,
        epsmin=np.finfo(type(X[0, 0])).eps, max_iter=max_iter,
        tol=1e-10, verbose=2, track_error=True,
        calculate_reconstruction_error_func=calculate_reconstruction_error_func,
        update_uv_batch_func=update_uv_batch_func,
        nchunkiter=1
    ), colors):
        for j in range(components):
            axs_comp[j].plot(U[:,j], color=color, alpha=0.5)
        for j in range(nexamples):
            axs_rec[j].plot(U @ V[:,j], color=color, alpha=0.25)
    fig_comp.savefig('test_normal_evolution.pdf')
    plt.close(fig_comp)
    fig_rec.savefig('test_normal_reconstruction_evolution.pdf')
    plt.close(fig_rec)


def test_random(plot=False):
    ## An example on simulated data
    n = 401
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

    init = 'random'
    ls = '-'
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
        ax_components.plot(fit.U + np.arange(components)[None,:], ls=ls)
        color = None
        for i, err_tracked in enumerate(fit.error_tracker):
            l, = ax_loss.plot(err_tracked, color=color,
                label=f'run {i} init {init}' if color is None else None, ls=ls)
            color = l.get_color()

    if plot:
        fig_components.savefig('test_normal.pdf')
        plt.close(fig_components)
        ax_loss.set_yscale('log')
        ax_loss.legend()
        fig_loss.savefig('test_normal_loss.pdf')
        plt.close(fig_loss)

    if plot:
        U_corr, V_corr = decorrelate_NMF_greedy(fit.U, fit.V, max_iter=100)
        fig_components2, ax_components2 = plt.subplots()
        ax_components2.plot(fit.U + np.arange(components)[None,:], color='gray')
        ax_components2.plot(U_corr + np.arange(components)[None,:])
        fig_components2.savefig('test_normal_corr.pdf')
        plt.close(fig_components2)

    # plot reconstructions
    if plot:
        plt.plot(X[:10].T, ls=' ', marker='o')
        plt.plot(fit.U @ fit.V[:,:10])
        plt.savefig('test_normal_reconstruct.pdf')
        plt.close()



    model.tol = 0
    V = model.transform(X=X, W=W)
    V2 = model.transform(X=X * 100, W=W / 100**2)
    assert_allclose(V * 100, V2, rtol=0.01)

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

    init = 'random'
    ls = '-'
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
        ax_components.plot(fit.U + np.arange(components)[None,:], ls=ls)
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
    elif sys.argv[1] == 'normal-evol':
        test_evolution(plot=True)
