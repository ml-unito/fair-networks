import theano
import numpy as np

sigma_init = 0.01
prng = np.random.RandomState(12345)


def change_random_seed(seed):
    global prng
    prng = np.random.RandomState(seed)


def randmat(dim1, dim2, name):
    return theano.shared(value=prng.normal(0, sigma_init, (dim1, dim2)).astype(theano.config.floatX), name=name)


def randtensor(shape, name):
    return theano.shared(value=prng.normal(0, sigma_init, shape).astype(theano.config.floatX), name=name)


def zerotensor(shape, name, broadcastable=None):
    if broadcastable is None:
        broadcastable = [False] * len(shape)
    return theano.shared(value=np.zeros(shape).astype(theano.config.floatX), name=name, broadcastable=broadcastable)


def zerovector(dim, name):
    return theano.shared(value=np.zeros((dim,)).astype(theano.config.floatX), name=name)


def append_hidden_layers(dim_in=50, dim_h=[50], normed=True, appended_name='_en'):
    """
    Append hidden layers to a neural network
        dim_in : The dimensionality of the first hidden layer
        dim_h : the dimensionality of the consequtive hidden layers
        normed : Whether to have a scaling factor for the weights
        appended_name : Add extra terms to the parameter name

    Return:
        [[W0, s0, h0], [W1, s1, h1], ...]
    """
    h_prev = dim_in
    params = []
    for i, h in enumerate(dim_h):
        W = randmat(h_prev, h, 'Wh' + str(i + 1) + appended_name)
        b = zerovector(h, 'bh' + str(i + 1) + appended_name)
        if normed:
            s = zerovector(h, 'sh' + str(i + 1) + appended_name)
            W.name += '_scaled'
            params.append([W, s, b])
        else:
            params.append([W, b])
        h_prev = h
    return params


def create_input_layer(dim_in=[784, 2], dim_h0=50, normed=True, appended_name='_en'):
    """
    Create the first layer which can have multiple inputs
        dim_in : Dimensionalities of each of the input variables
        dim_h0 : Dimensionality of the output
        normed : Whether to have a scaling factor for the weights
        appended_name : Add extra terms to the parameter name

     Return:
        [[Win1, sin1], [Win2, sin2], ..., [bin]]
    """
    params = []
    for i, din in enumerate(dim_in):
        if isinstance(din, list):
            d_params = []
            for j in xrange(len(din)):
                W = randmat(
                    din[j], dim_h0, 'W' + str(j) + 'in' + str(i + 1) + appended_name)
                if normed:
                    s = zerovector(
                        dim_h0, 's' + str(j) + 'in' + str(i + 1) + appended_name)
                    W.name += '_scaled'
                    d_params.append([W, s])
                else:
                    d_params.append([W])
            params.append(d_params)
        else:
            W = randmat(din, dim_h0, 'Win' + str(i + 1) + appended_name)
            if normed:
                s = zerovector(dim_h0, 'sin' + str(i + 1) + appended_name)
                W.name += '_scaled'
                params.append([W, s])
            else:
                params.append([W])
    params.append([zerovector(dim_h0, 'bin' + appended_name)])
    return params


def create_output_layer(dim_in=50, dim_out=[50, 50], normed=True, appended_name='_en'):
    """
    Create the last layer which can have multiple outputs
        dim_in : Dimensionalities of each of the input variables
        dim_h0 : Dimensionality of the output
        normed : Whether to have a scaling factor for the weights
        appended_name : Add extra terms to the parameter name

     Return:
        [[Wout1, sout1, bout1], [Wout2, sout2, bout2]]
    """
    params = []
    for i, dout in enumerate(dim_out):
        if isinstance(dout, list):
            d_params = []
            for j in xrange(len(dout)):
                W = randmat(
                    dim_in, dout[j], 'W' + str(j) + 'out' + str(i + 1) + appended_name)
                b = zerovector(
                    dout[j], 'b' + str(j) + 'out' + str(i + 1) + appended_name)
                if normed:
                    s = zerovector(
                        dout[j], 's' + str(j) + 'out' + str(i + 1) + appended_name)
                    W.name += '_scaled'
                    d_params.append([W, s, b])
                else:
                    d_params.append([W, b])
            params.append(d_params)
        else:
            W = randmat(dim_in, dout, 'Wout' + str(i + 1) + appended_name)
            b = zerovector(dout, 'bout' + str(i + 1) + appended_name)
            if normed:
                s = zerovector(dout, 'sout' + str(i + 1) + appended_name)
                W.name += 'scaled'
                params.append([W, s, b])
            else:
                params.append([W, b])
    return params
