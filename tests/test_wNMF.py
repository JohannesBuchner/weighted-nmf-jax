import numpy as np
from wNMF import wNMF
import matplotlib.pyplot as plt

def test_random():
    ## An example on simulated data
    n = 101
    features = 100
    components = 4
    
    shapes_true = np.array([0.5 + 0.5 * np.sin(np.arange(features) / 10 / 2**i + np.random.uniform(0, np.pi)) for i in range(components)])
    shapes_true[0] = 1
    shapes_true[1] = np.exp(-np.arange(features) / 400)

    plt.plot(shapes_true.T);
    plt.savefig('test_normal_input.pdf')
    plt.close()

    X = np.random.normal(100 * np.random.uniform(size=(n, components)) @ shapes_true, 1)
    W = np.ones_like(X)
    plt.plot(X[:10].T, ls=' ', marker='o');
    plt.savefig('test_normal_data.pdf')
    plt.close()

    model = wNMF(n_components=components, beta_loss='frobenius', max_iter=1000, track_error=True, verbose=2)
    fit = model.fit(X=X, W=W, n_run=5)
    print(fit.V.shape)
    print(fit.U.shape)
    assert fit.V.shape == (components, features)
    assert fit.U.shape == (n, components)
    assert np.shape(fit.err) == ()
    assert np.shape(fit.err) == ()
    assert len(fit.error_tracker) == 5
    assert len(fit.error_tracker[0]) == 1000

    plt.plot(fit.V.T);
    plt.savefig('test_normal.pdf')
    plt.close()

    for i, err_tracked in enumerate(fit.error_tracker):
        plt.plot(err_tracked, label=f'run {i}');
    plt.yscale('log')
    plt.legend()
    plt.savefig('test_normal_loss.pdf')
    plt.close()

def test_poisson():
    ## An example on simulated data
    n = 101
    features = 1000
    components = 4
    n_run = 1
    np.random.seed(1)

    shapes_true = np.array([0.5 + 0.5 * np.sin(np.arange(features) / 10 / 2**i + np.random.uniform(0, np.pi)) for i in range(components)])
    shapes_true[0] = 1
    shapes_true[1] = np.exp(-np.arange(features) / 400)
    shapes_true /= shapes_true.max(axis=0, keepdims=True)

    for k, component in enumerate(shapes_true):
        plt.plot(component, label=f'component {k}')
    plt.legend()
    plt.savefig('test_poisson_input.pdf')
    plt.close()

    ## An example on simulated data
    X = 1. * np.random.poisson(100 * (10**np.random.uniform(-4, 2, size=(n, components))) @ shapes_true)
    W = np.ones_like(X)
    plt.plot(X[:10].T, ls=' ', marker='o');
    plt.savefig('test_poisson_data.pdf')
    plt.close()

    model = wNMF(n_components=components, beta_loss='kullback-leibler', max_iter=1000, track_error=True, verbose=2)
    fit = model.fit(X=X, W=W, n_run=n_run)

    assert fit.V.shape == (components, features)
    assert fit.U.shape == (n, components)
    assert np.shape(fit.err) == ()
    assert np.shape(fit.err) == ()
    assert len(fit.error_tracker) == n_run
    assert len(fit.error_tracker[0]) == 1000

    for k, component in enumerate(fit.V):
        plt.plot(component, label=f'component {k}')
    plt.legend()
    plt.savefig('test_poisson.pdf')
    plt.close()

    for i, err_tracked in enumerate(fit.error_tracker):
        plt.plot(err_tracked, label=f'run {i}', lw=5 if np.argmin(fit.err_all) == i else 1);
    plt.yscale('log')
    plt.legend()
    plt.savefig('test_poisson_loss.pdf')
    plt.close()

if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'poisson':
        test_poisson()
    else:
        test_random()
