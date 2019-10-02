import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
import sys


class BaseOptimizer(object):

    """
    Base optimizer class
    """

    def __init__(self, objectives, params, gradients=None, regularization='l2', normalization=None,
                 batch_size=128, polyak=True, beta3=0.1):
        self.normalization = normalization
        self.regularization = regularization
        self.batch_size = batch_size
        self.batches_test = 1
        self.polyak = polyak
        self.beta3 = beta3

        if gradients is None:
            self.gradients = T.grad(T.sum(objectives), params, disconnected_inputs='warn', add_names=True)
        else:
            self.gradients = gradients
            if regularization is not None:
                print('Warning! You already passed gradients and a regularizer. ' \
                      'Make sure that the regulizer is not already precomputed.')

    def train(self, data, verbose=False):
        batches = np.arange(0, data[0].shape[0], self.batch_size)
        lb = 0
        rounding = lambda x: ['%.3f' % i for i in x]
        for j in range(len(batches) - 1):
            inp = [d[batches[j]:batches[j + 1]] for d in data]
            objectives = np.array(self._ascent(*inp))
            self._update_inf()
            if np.isnan(objectives).any():
                print(lb)
                print(objectives)
                raise Exception('NaN objective!')
            lb += objectives
            if verbose:
                sys.stdout.write("\rBatch:{0}, Objectives:{1}, Total:{2}".format(
                    str(j + 1) + '/' + str(len(batches) - 1), str(rounding((objectives).tolist())), str(rounding(lb.tolist()))))
                sys.stdout.flush()
        if verbose:
            print()

        return lb

    def evaluate(self, data, batch_test=False, verbose=False):
        if self.batches_test == 1 and not batch_test:
            return np.array(self._eval(*data))

        batches = np.arange(0, data[0].shape[0], self.batch_size)
        rounding = lambda x: ['%.3f' % i for i in x]
        lb = 0
        for j in range(len(batches) - 1):
            inp = [d[batches[j]:batches[j + 1]] for d in data]
            objectives = np.array(self._eval(*inp))
            lb += objectives
            if verbose:
                sys.stdout.write("\rEval_Batch:{0}, Objectives:{1}, Total:{2}".format(
                    str(j + 1) + '/' + str(len(batches) - 1), str(rounding((objectives).tolist())), str(rounding(lb.tolist()))))
                sys.stdout.flush()
        if verbose:
            print()
        return lb

    def _ascent(self):
        raise NotImplementedError()

    def _eval(self):
        raise NotImplementedError()

    def _update_inf(self):
        raise NotImplementedError()

    def normalize_param(self, param, w_):
        if '_scaled' in param.name and self.normalization is not None:
            # print self.norm, 'normalization on', params[i].name
            if self.normalization == 'l2':
                w_new = w_ / T.sqrt(T.sum(T.sqr(w_), axis=0, keepdims=True))
            elif self.normalization == 'l1':
                w_new = w_ / T.sum(T.abs_(w_), axis=0, keepdims=True)
            elif self.normalization == 'nonorm':
                w_new = w_
            else:
                raise NotImplementedError()
        else:
            w_new = w_

        return w_new

    def get_updates_eval(self, params_inf, params):
        """
        Keep an exponential moving average of the parameters, which will be used for evaluation
        """
        updates_eval = OrderedDict()

        itinf = theano.shared(0., name='itinf')
        updates_eval[itinf] = itinf + 1.
        fix3 = 1. - self.beta3**(itinf + 1.)

        for i in range(len(params)):
            if self.polyak:
                if 'scalar' in params_inf[i].name:
                    avg = theano.shared(np.float32(0.), name=params_inf[i].name + '_avg')
                else:
                    avg = theano.shared(params_inf[i].get_value() * 0., name=params_inf[i].name + '_avg',
                                        broadcastable=params_inf[i].broadcastable)

                avg_new = self.beta3 * avg + (1. - self.beta3) * params[i]
                updates_eval[avg] = T.cast(avg_new, theano.config.floatX)
                updates_eval[params_inf[i]] = T.cast(avg_new / fix3, theano.config.floatX)
            else:
                updates_eval[params_inf[i]] = params[i]

        return updates_eval


class AdaM(BaseOptimizer):

    """
    AdaM optimizer for an objective function
    """

    def __init__(self, objectives, objectives_eval, inputs, params, params_inf, gradients=None, regularization='l2',
                 normalization=None, weight_decay=0., alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=128,
                 polyak=True, beta3=0.9, **kwargs):

        super(AdaM, self).__init__(objectives, params, gradients=gradients,
                                   regularization=regularization, normalization=normalization,
                                   batch_size=batch_size, polyak=polyak, beta3=beta3)

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        updates = self.get_updates(params, self.gradients)
        updates_eval = self.get_updates_eval(params_inf, params)
        inputs = inputs

        # evaluate all the objectives and update parameters
        self._ascent = theano.function(inputs, objectives, updates=updates, on_unused_input='ignore', mode='FAST_RUN')
        # evaluate all the objectives and (optionally) use a moving average for the parameters
        self._update_inf = theano.function([], [], updates=updates_eval, on_unused_input='ignore', mode='FAST_RUN')
        self._eval = theano.function(inputs, objectives_eval, on_unused_input='ignore', mode='FAST_RUN')
        print('AdaM', 'alpha:', alpha, 'beta1:', beta1, 'beta2:', beta2, 'epsilon:', self.epsilon, \
            'batch_size:', self.batch_size, 'normalization:', normalization, 'regularization:', regularization, \
            'weight_decay:', weight_decay, 'polyak:', polyak, 'beta3:', beta3)

    def get_updates(self, params, grads):
        updates = OrderedDict()

        it = theano.shared(0., name='it')
        updates[it] = it + 1.
        lr = self.alpha

        fix1 = 1. - self.beta1**(it + 1.)  # To make estimates unbiased
        fix2 = 1. - self.beta2**(it + 1.)  # To make estimates unbiased

        for i in range(len(grads)):
            gi = grads[i]
            if self.regularization is not None:
                if self.regularization == 'l1':
                    gi -= self.weight_decay * T.sgn(params[i])
                elif self.regularization == 'l2':
                    gi -= self.weight_decay * params[i]

            # mean_squared_grad := E[g^2]_{t-1}
            if 'scalar' in params[i].name:
                mom1 = theano.shared(np.float32(0.))
                mom2 = theano.shared(np.float32(0.))
            else:
                mom1 = theano.shared(params[i].get_value() * 0., broadcastable=params[i].broadcastable)
                mom2 = theano.shared(params[i].get_value() * 0., broadcastable=params[i].broadcastable)

            # Update moments
            mom1_new = self.beta1 * mom1 + (1. - self.beta1) * gi
            mom2_new = self.beta2 * mom2 + (1. - self.beta2) * T.sqr(gi)

            # Compute the effective gradient
            corr_mom1 = mom1_new / fix1
            corr_mom2 = mom2_new / fix2
            effgrad = corr_mom1 / (T.sqrt(corr_mom2) + self.epsilon)

            # Do update
            w_ = params[i] + lr * effgrad
            # Apply normalization
            w_new = self.normalize_param(params[i], w_)

            # Apply update
            updates[params[i]] = T.cast(w_new, theano.config.floatX)
            updates[mom1] = T.cast(mom1_new, theano.config.floatX)
            updates[mom2] = T.cast(mom2_new, theano.config.floatX)

        return updates


class AdaMax(BaseOptimizer):

    """
    AdaMax optimizer for an objective function (variant of Adam based on infinity norm)
    """

    def __init__(self, objectives, objectives_eval, inputs, params, params_inf, gradients=None, regularization='l2',
                 normalization='l2', weight_decay=0., alpha=0.002, beta1=0.9, beta2=0.999, batch_size=128, polyak=True,
                 beta3=0.9, **kwargs):

        super(AdaMax, self).__init__(objectives, params, gradients=gradients,
                                     regularization=regularization, normalization=normalization,
                                     batch_size=batch_size, polyak=polyak, beta3=beta3)

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        updates = self.get_updates(params, self.gradients)
        updates_eval = self.get_updates_eval(params_inf, params)

        # evaluate all the objectives and update parameters
        self._ascent = theano.function(inputs, objectives, updates=updates, on_unused_input='ignore', mode='FAST_RUN')
        # evaluate all the objectives and update evaluation parameters
        self._update_inf = theano.function([], [], updates=updates_eval, on_unused_input='ignore', mode='FAST_RUN')
        # evaluate all the objectives and do not update the parameters
        self._eval = theano.function(inputs, objectives_eval, on_unused_input='ignore', mode='FAST_RUN')
        print('AdaMax', 'alpha:', alpha, 'beta1:', beta1, 'beta2:', beta2, 'batch_size:', self.batch_size, \
            'normalization:', normalization, 'regularization:', regularization, 'weight_decay:', weight_decay, \
            'polyak:', polyak, 'beta3:', beta3)

    def get_updates(self, params, grads):
        updates = OrderedDict()

        it = theano.shared(0., name='it')
        updates[it] = it + 1.

        for i in range(len(grads)):
            gi = grads[i]
            if self.regularization is not None:
                if self.regularization == 'l1':
                    gi -= self.weight_decay * T.sgn(params[i])
                elif self.regularization == 'l2':
                    gi -= self.weight_decay * params[i]

            # mean_squared_grad := E[g^2]_{t-1}
            mom1 = theano.shared(
                params[i].get_value() * 0., broadcastable=params[i].broadcastable)
            mom2 = theano.shared(
                params[i].get_value() * 0., broadcastable=params[i].broadcastable)

            # Update moments
            mom1_new = self.beta1 * mom1 + (1. - self.beta1) * gi
            mom2_new = T.maximum(self.beta2 * mom2, np.abs(gi))

            # Compute the effective gradient
            # To make estimates unbiased
            corr_mom1 = mom1_new / (1. - self.beta1 ** (it + 1.))
            effgrad = corr_mom1 / mom2_new

            # Do update
            w_ = params[i] + self.alpha * effgrad
            # Apply normalization
            w_new = self.normalize_param(params[i], w_)

            # Apply update
            updates[params[i]] = T.cast(w_new, theano.config.floatX)
            updates[mom1] = T.cast(mom1_new, theano.config.floatX)
            updates[mom2] = T.cast(mom2_new, theano.config.floatX)

        return updates
