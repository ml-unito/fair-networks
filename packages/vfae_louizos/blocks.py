import theano
import theano.tensor as T
import numpy as np

srng = T.shared_randomstreams.RandomStreams(seed=12345)
# prng = np.random.RandomState(12345)
low = 3
high = 8


def change_random_seed(seed):
    global srng
    srng = T.shared_randomstreams.RandomStreams(seed=seed)

'''
Nonlinear functions
'''
leaky_relu = lambda x, alpha=0.3: T.nnet.relu(x, alpha=alpha)
softsign = lambda x: x / (1. + np.abs(x))
linear = lambda x: x
nonlinearities = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softmax': T.nnet.softmax, 'softplus': T.nnet.softplus,
                  'relu': T.nnet.relu, 'leaky_relu': leaky_relu, 'linear': linear, 'softsign': softsign}

'''
Weight normalization functions
'''
l2norm = lambda W: W / T.sqrt(T.sum(T.sqr(W), axis=0, keepdims=True))
l1norm = lambda W: W / T.sum(T.abs_(W), axis=0, keepdims=True)
normalizations = {'l2': l2norm, 'l1': l1norm}

'''
Kernel functions
'''
rbf = lambda x1, x2, gamma=1.: T.exp(
    -((x1[np.newaxis, :, :] - x2[:, np.newaxis, :]) ** 2).sum(2) * gamma).T
poly = lambda x1, x2, degree=2, gamma=1., bias=1.: (
    (gamma * T.dot(x1, x2.T)) + bias)**degree
identity = lambda x1, x2: T.sum((T.mean(x1, axis=0) - T.mean(x2, axis=0))**2)
def mmd_fourier(x1, x2, bandwidth=2., dim_r=500):
    """
    Approximate RBF kernel by random features
    """
    rW_n = T.sqrt(2. / bandwidth) * srng.normal((x1.shape[1], dim_r)).astype(theano.config.floatX) / T.sqrt(x1.shape[1])
    rb_u = 2 * np.pi * srng.uniform((dim_r,)).astype(theano.config.floatX)
    rf0 = T.sqrt(2. / rW_n.shape[1]) * T.cos(x1.dot(rW_n) + rb_u)
    rf1 = T.sqrt(2. / rW_n.shape[1]) * T.cos(x2.dot(rW_n) + rb_u)
    return ((rf0.mean(0) - rf1.mean(0))**2).sum()
kernels = {'rbf': rbf, 'poly': poly, 'identity': identity, 'rbf_fourier': mmd_fourier}


def mmd_objective(x1, x2, kernel='rbf', bandwidths=1. / (2 * (np.array([1., 2., 5., 8., 10])**2))):
    """
    Return the mmd score between a pair of observations
    """
    K = kernels[kernel]
    if kernel == 'identity':
        return T.sqrt(K(x1, x2))
    elif kernel == 'rbf_fourier':
        return T.sqrt(K(x1, x2, bandwidth=2.))

    # possibly mixture of kernels
    x1x1, x1x2, x2x2 = 0, 0, 0
    for bandwidth in bandwidths:
        x1x1 += K(x1, x1, gamma=T.sqrt(x1.shape[1]) * bandwidth) / len(bandwidths)
        x2x2 += K(x2, x2, gamma=T.sqrt(x2.shape[1]) * bandwidth) / len(bandwidths)
        x1x2 += K(x1, x2, gamma=T.sqrt(x1.shape[1]) * bandwidth) / len(bandwidths)

    return T.sqrt(x1x1.mean() - 2 * x1x2.mean() + x2x2.mean())


class MLP(object):

    def __init__(self, input_params, hidden_layers, nonlin='softplus', normalization='l2', dropout_rate=0.):
        """
        Deterministic MLP that takes a number of inputs and returns the last hidden layer

        inputs: List of inputs
            [x0, x1, ...]
        input_params : List of parameters for the inputs
            [[W0, s0], [W1, s1], ...., [b]]
        hidden_layers : List of hidden layer params
            [[W_h1, s_h1, b_h1], [W_h2, s_h2, b_h2], ...]
        nonlin : nonlinearity to use
        """
        self.input_params = input_params
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.sigmoid = nonlinearities['sigmoid']
        self.nonlin = nonlin

        # get the corresponding activation
        self.activation = nonlinearities[nonlin]
        if normalization is not None:
            self.normalize = normalizations[normalization]

    def ff_inputs(self, inputs, input_params=None, inference=False):
        """
        Feedforward just the inputs
        """
        if input_params is None:
            input_params = self.input_params
        # do the dot product of the inputs
        lin_dot = T.as_tensor_variable(0)
        drop_rate = max(0., self.dropout_rate - 0.3)   # lower dropout rate for input layer
        for inp, params in zip(inputs, input_params[:-1]):
            W = self.normalize(params[0]) * T.exp(params[1]) if len(params) > 1 else params[0]
            if inp.name is not None:
                if 'nuisance_' not in inp.name:  # do dropout only on the non-nuisance variables (since they are 1ofk encoded)
                    if drop_rate > 0:
                        if not inference:
                            inp *= srng.binomial(size=inp.shape, n=1, p=(1. - drop_rate)).astype(
                                theano.config.floatX) / (1. - drop_rate)
                    lin_dot += T.dot(inp, W)
                else:
                    lin_dot += T.dot(inp, W)
            else:
                if drop_rate > 0:
                    if not inference:
                        inp *= srng.binomial(size=inp.shape, n=1, p=(1. - drop_rate)).astype(
                            theano.config.floatX) / (1. - drop_rate)
                lin_dot += T.dot(inp, W)

        # add the bias
        lin_dot += input_params[-1][0]
        h = [self.activation(lin_dot)]
        # return output
        return h

    def ff_hidden_layers(self, h, start=0, inference=False):
        """
        Continue to the remaining hidden layers
        """
        # feedforward from the remaining layers
        if len(self.hidden_layers[start:]) > 0:
            for i, layer in enumerate(self.hidden_layers):
                W = self.normalize(layer[0]) * T.exp(layer[1]) if len(layer) > 2 else layer[0]
                hin = h[-1]
                if not inference:
                    if self.dropout_rate > 0:
                        hin *= srng.binomial(size=hin.shape, n=1, p=(1. - self.dropout_rate)).astype(
                            theano.config.floatX) / (1. - self.dropout_rate)
                h_ = self.activation(T.dot(hin, W) + layer[-1])
                h.append(h_)
        # return output
        return h

    def ff(self, inputs, inference=False):
        """
        Wrapper for the full feedforward process
        """
        h = self.ff_inputs(inputs, inference=inference)
        h2 = self.ff_hidden_layers(h, inference=inference)
        return h2


class DiagGaussianEncoder(MLP):
    """
    Encoder that maps inputs to a latent Gaussian distribution
    """
    def __init__(self, input_params, hidden_layers, latent_layer, batch_size=1, nonlin='softplus',
                 normalization='l2', dropout_rate=0., prior_mu=0., prior_sg=1.):

        super(DiagGaussianEncoder, self).__init__(input_params, hidden_layers, nonlin=nonlin,
                                                  normalization=normalization, dropout_rate=dropout_rate)
        self.latent_layer = latent_layer
        self.batch_size = batch_size
        if normalization is not None:
            self.normalize = normalizations[normalization]
        self.prior_mu = prior_mu
        self.prior_sg = prior_sg

    def transform(self, inputs, constrain_means=False, inference=False):
        # ff from the deterministic MLP
        hf = self.ff(inputs, inference=inference)
        hout = hf[-1]
        if self.dropout_rate > 0:
            if not inference:
                hout *= srng.binomial(size=hout.shape, n=1, p=(1. - self.dropout_rate)).astype(
                    theano.config.floatX) / (1. - self.dropout_rate)
        # get the weights for the mean and logvariance
        Wmu = self.normalize(self.latent_layer[0][0]) * T.exp(self.latent_layer[0][1]) \
            if len(self.latent_layer[0]) > 2 else self.latent_layer[0][0]
        Wstd = self.normalize(self.latent_layer[1][0]) * T.exp(self.latent_layer[1][1]) \
            if len(self.latent_layer[1]) > 2 else self.latent_layer[1][0]

        # estimate the parameters
        mu_z = T.dot(hout, Wmu) + self.latent_layer[0][-1]
        if constrain_means:
            mu_z = T.nnet.sigmoid(mu_z)
        std_z = T.nnet.softplus(T.dot(hout, Wstd) + self.latent_layer[1][-1])
        return mu_z, std_z

    def sample(self, mu_z, std_z):
        return mu_z + std_z * srng.normal(mu_z.shape).astype(theano.config.floatX)

    def kldivergence(self, mu_z, std_z):
        return self.kldivergence_perx(mu_z, std_z).sum()

    def kldivergence_perx(self, mu_z, std_z):
        return 0.5 * T.sum(1 - T.log(self.prior_sg**2) + T.log(std_z**2) -
                           ((mu_z - self.prior_mu)**2 + std_z**2) / self.prior_sg**2, axis=1)

    def kldivergence_givenp(self, mu_zq, std_zq, mu_zp, std_zp):
        return self.kldivergence_givenp_perx(mu_zq, std_zq, mu_zp, std_zp).sum()

    def kldivergence_givenp_perx(self, mu_zq, std_zq, mu_zp, std_zp):
        return 0.5 * T.sum(1 - T.log(std_zp**2) + T.log(std_zq**2) -
                           ((mu_zq - mu_zp)**2 + std_zq**2) / std_zp**2, axis=1)

    def logp_perx(self, sample):
        return -.5 * (T.log(2 * np.pi) + T.log(self.prior_sg**2) + (sample - self.prior_mu)**2 / self.prior_sg**2).sum(axis=1)

    def logp(self, sample):
        return T.sum(self.logp_perx(sample))

    def logq_perx(self, sample, mu_z, std_z):
        return -.5 * (T.log(2 * np.pi) + T.log(std_z**2) + ((sample - mu_z)**2) / (std_z**2)).sum(axis=1)

    def logq(self, sample, mu_z, std_z):
        return T.sum(self.logq_perx(sample, mu_z, std_z))


class DiagGaussianDecoder(MLP):
    """
    Stochastic decoder that maps inputs to a Gaussian distribution with diagonal covariance
    """
    def __init__(self, input_params, hidden_layers, reconstruction_layer, nonlin='softplus', normalization='l2',
                 dropout_rate=0.):

        super(DiagGaussianDecoder, self).__init__(input_params, hidden_layers, nonlin=nonlin,
                                                  normalization=normalization, dropout_rate=dropout_rate)
        self.reconstruction_layer = reconstruction_layer
        if normalization is not None:
            self.normalize = normalizations[normalization]

    def transform(self, inputs, constrain_means=False, inference=False, mult=1., add=0.):
        # ff from the deterministic mlp
        if len(self.input_params) > 0:   # if we do not have any hidden layers
            hf = self.ff(inputs, inference=inference)
            hout = hf[-1]
            if self.dropout_rate > 0:
                if not inference:
                    hout *= srng.binomial(size=hout.shape, n=1, p=(1. - self.dropout_rate)).astype(
                        theano.config.floatX) / (1. - self.dropout_rate)
        else:
            hout = inputs[0]
            if self.dropout_rate > 0:
                drop_rate = max(0., self.dropout_rate - 0.3)
                if not inference:
                    hout *= srng.binomial(size=hout.shape, n=1, p=(1. - self.dropout_rate)).astype(
                        theano.config.floatX) / (1. - drop_rate)

        # get the weights for the mean and logvariance
        Wmu = self.normalize(self.reconstruction_layer[0][0]) * T.exp(self.reconstruction_layer[0][1]) \
            if len(self.reconstruction_layer[0]) > 2 else self.reconstruction_layer[0][0]
        Wstd = self.normalize(self.reconstruction_layer[1][0]) * T.exp(self.reconstruction_layer[1][1]) \
            if len(self.reconstruction_layer[1]) > 2 else self.reconstruction_layer[1][0]
        # get the mean and variances of logp(x | inputs)
        mu_x = T.dot(hout, Wmu) + self.reconstruction_layer[0][-1]
        if constrain_means:
            mu_x = mult * T.nnet.sigmoid(mu_x) + add
        std_x = T.nnet.softplus(T.dot(hout, Wstd) + self.reconstruction_layer[1][-1])

        return mu_x, std_x

    def logp(self, x, mu_x, std_x):
        return T.sum(self.logp_perx(x, mu_x, std_x))

    def logp_perx(self, x, mu_x, std_x):
        return -.5 * (T.log(2 * np.pi) + T.log(std_x**2) + ((x - mu_x)**2) / (std_x**2)).sum(axis=1)

    def kldivergence(self, mu_x, std_x, prior_mu_x, prior_std_x):
        return T.sum(self.kldivergence_perx(mu_x, std_x, prior_mu_x, prior_std_x))

    def kldivergence_perx(self, mu_x, std_x, prior_mu_x, prior_std_x):
        return 0.5 * T.sum(1 - T.log(prior_std_x**2) + T.log(std_x**2) -
                        ((mu_x - prior_mu_x)**2 + std_x**2) / prior_std_x**2, axis=1)

    def sample(self, mu_x, std_x):
        return mu_x + std_x * srng.normal(mu_x.shape).astype(theano.config.floatX)


class BernoulliDecoder(MLP):
    """
    Stochastic decoder that maps to a bernoulli distribution
    """
    def __init__(self, input_params, hidden_layers, reconstruction_layer, nonlin='softplus', normalization='l2',
                 dropout_rate=0.):
        super(BernoulliDecoder, self).__init__(input_params, hidden_layers, nonlin=nonlin, normalization=normalization,
                                               dropout_rate=dropout_rate)
        self.reconstruction_layer = reconstruction_layer
        if normalization is not None:
            self.normalize = normalizations[normalization]

    def transform(self, inputs, inference=False):
        hf = self.ff(inputs, inference=inference)
        hout = hf[-1]
        if self.dropout_rate > 0:
            if not inference:
                hout *= srng.binomial(size=hout.shape, n=1, p=(1. - self.dropout_rate)).astype(
                    theano.config.floatX) / (1. - self.dropout_rate)

        # get the mean of the bernoulli
        Wp = self.normalize(self.reconstruction_layer[0][0]) * T.exp(self.reconstruction_layer[0][1]) \
            if len(self.reconstruction_layer[0]) > 2 else self.reconstruction_layer[0][0]
        p = T.nnet.sigmoid(T.dot(hout, Wp) + self.reconstruction_layer[0][-1])
        return [p]

    def logp(self, x, p):
        return T.sum(self.logp_perx(x, p))

    def logp_perx(self, x, p):
        return -T.nnet.binary_crossentropy(p, x).sum(axis=1)

    def sample(self, p):
        return srng.binomial(size=p.shape, n=1, p=p).astype(theano.config.floatX)


class PoissonDecoder(MLP):
    """
    Stochastic decoder that maps to a poisson distribution
    """
    def __init__(self, input_params, hidden_layers, reconstruction_layer, nonlin='softplus', normalization='l2',
                 dropout_rate=0.):
        super(PoissonDecoder, self).__init__(input_params, hidden_layers, nonlin=nonlin, normalization=normalization,
                                             dropout_rate=dropout_rate)
        self.reconstruction_layer = reconstruction_layer
        if normalization is not None:
            self.normalize = normalizations[normalization]

    def transform(self, inputs, inference=False):
        # ff through the MLP
        hf = self.ff(inputs, inference=inference)
        hout = hf[-1]
        if self.dropout_rate > 0:
            if not inference:
                hout *= srng.binomial(size=hout.shape, n=1, p=(1. - self.dropout_rate)).astype(
                    theano.config.floatX) / (1. - self.dropout_rate)

        # get the log(mean/variance) of the Poisson
        Wloglambda = self.normalize(self.reconstruction_layer[0][0]) * T.exp(self.reconstruction_layer[0][1]) \
            if len(self.reconstruction_layer[0]) > 2 else self.reconstruction_layer[0][0]
        lamba = T.nnet.softplus(T.dot(hout, Wloglambda) + self.reconstruction_layer[0][-1])
        return [lamba]

    def logp(self, x, lamba):
        return T.sum(self.logp_perx(x, lamba))

    def logp_perx(self, x, lamba):
        # return T.sum((-lamba + x*T.log(lamba) - T.gammaln(x +
        # 1)).sum(axis=1))
        # logGamma(x+1) is constant w.r.t. the optimization
        return (-lamba + x * T.log(lamba)).sum(axis=1)

    def sample(self, lamba):
        return srng.poisson(size=lamba.shape, lam=lamba, dtype='int64')


class CategoricalDecoder(MLP):

    def __init__(self, input_params, hidden_layers, reconstruction_layer, nonlin='softplus', normalization='l2',
                 dropout_rate=0.):
        super(CategoricalDecoder, self).__init__(input_params, hidden_layers,
                                                 nonlin=nonlin, normalization=normalization, dropout_rate=dropout_rate)
        self.reconstruction_layer = reconstruction_layer
        if normalization is not None:
            self.normalize = normalizations[normalization]
        self.softmax = nonlinearities['softmax']

    def transform(self, inputs, inference=False):
        # ff from the deterministic mlp
        if len(self.input_params) > 0:
            hf = self.ff(inputs, inference=inference)
            hout = hf[-1]
            if self.dropout_rate > 0:
                if not inference:
                    hout *= srng.binomial(size=hout.shape, n=1, p=(1. - self.dropout_rate)).astype(
                        theano.config.floatX) / (1. - self.dropout_rate)
        else:  # if we do not have any hidden layers
            hout = inputs[0]
            if self.dropout_rate > 0:
                drop_rate = min(0., self.dropout_rate - 0.3)
                if not inference:
                    hout *= srng.binomial(size=hout.shape, n=1, p=(1. - drop_rate)).astype(
                        theano.config.floatX) / (1. - drop_rate)

        W = self.normalize(self.reconstruction_layer[0][0]) * T.exp(self.reconstruction_layer[0][1]) \
            if len(self.reconstruction_layer[0]) > 2 else self.reconstruction_layer[0][0]
        # get the probabilities of each category
        ps = self.softmax(T.dot(hout, W) + self.reconstruction_layer[0][-1])
        return [ps]

    def logp(self, x, ps):
        return T.sum(self.logp_perx(x, ps))

    def logp_perx(self, x, ps):
        return T.log(ps)[T.arange(x.shape[0]), x]

    def entr(self, ps):
        return - T.sum(ps * T.log(ps), axis=1).sum()

    def kldivergence(self, prior, ps):
        return - T.sum(ps * (T.log(prior) - T.log(ps)), axis=1).sum()

    def most_probable(self, ps):
        return T.argmax(ps, axis=1)

    def sample(self, ps):
        return srng.multinomial(size=ps.shape[0], n=1, pvals=ps).astype(theano.config.floatX)
