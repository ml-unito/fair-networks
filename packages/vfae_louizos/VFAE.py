import theano
import theano.tensor as T
import vfae_louizos.blocks as blk
from vfae_louizos.optimizer import AdaM, AdaMax
import vfae_louizos.generate_params as gpm
import time
import numpy as np
import os


def log_f(file_, string):
    print(string)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    with open('logs/' + file_, "a") as myfile:
        myfile.write("\n" + string)


class VFAE(object):

    """
    Variational Fair autoencoder that implements a generative model with two layers of stochastic variables,
    where both are conditional, i.e.:
        p(x, z1, z2, y | s) = p(z2)p(y)p(z1|z2,y)p(x|z1, s)
    with q(z1|x,s)q(z2|z1,y)q(y|z1) being the variational posteriors.

    Furthermore there is an extra MMD penalty on z1 to further enforce independence between z1 and s.
    """

    def __init__(self, N, dim_x, dim_s, dim_y, dim_h_en_z1=(50, 50), dim_h_en_z2=(50, 50), dim_h_de_z1=(50, 50),
                 dim_h_de_x=(50, 50), dim_h_clf=(50, 50), dim_z1=50, dim_z2=50, batch_size=100, nonlinearity='softplus',
                 normalization='l2', regularization='l2', type_rec='binary', iterations=500, use_MMD=True,
                 kernel_MMD='rbf_fourier', lambda_reg=1., supervised_rate=1., constrain_means=False, dropout_rate=0.,
                 weight_decay=None, learningRate=0.001, mode='FAST_RUN', prior_y='uniform', use_s=True,
                 type_y='discrete', semi_supervised=False, log_txt='', optim_alg='adam', random_seed=12345, beta1=0.9,
                 beta2=0.999, polyak=False, beta3=0.9, L=1):
        self.dim_x = dim_x
        self.dim_s = dim_s
        self.dim_y = dim_y
        self.dim_h_en_z1 = dim_h_en_z1
        self.dim_h_de_z1 = dim_h_de_z1
        self.dim_h_en_z2 = dim_h_en_z2
        self.dim_h_de_x = dim_h_de_x
        self.dim_h_clf = dim_h_clf
        self.dim_z1 = dim_z1
        self.dim_z2 = dim_z2
        self.batch_size = batch_size
        self.nonlinearity = nonlinearity
        self.normalization = normalization
        self.regularization = regularization
        self.learningRate = learningRate
        self.normed = False
        if self.normalization is not None:
            self.normed = True
        if weight_decay is None and regularization is not None:
            self.weight_decay = (1. * batch_size) / N
        elif regularization is not None:
            self.weight_decay = weight_decay
        else:
            self.weight_decay = 0.
        self.beta1 = beta1
        self.beta2 = beta2
        self.polyak = polyak
        self.beta3 = beta3
        self.type_rec = type_rec
        if self.type_rec == 'binary':
            self.decoder = blk.BernoulliDecoder
        elif self.type_rec == 'diag_gaussian':
            self.decoder = blk.DiagGaussianDecoder
        elif self.type_rec == 'poisson':
            self.decoder = blk.PoissonDecoder
        else:
            raise Exception()

        self.encoder = blk.DiagGaussianEncoder
        self.decoder_z1_ = blk.DiagGaussianDecoder
        self.L = L

        self.iterations = iterations
        self.use_MMD = use_MMD
        self.kernel_MMD = kernel_MMD
        self.lambda_reg = lambda_reg
        self.supervised_rate = supervised_rate
        self.constrain_means = constrain_means
        self.dropout_rate = dropout_rate
        self.mode = mode
        self.prior_y = prior_y
        self.use_s = use_s
        self.type_y = type_y
        self.semi_supervised = semi_supervised
        self.log_txt = log_txt

        blk.change_random_seed(random_seed)
        gpm.change_random_seed(random_seed)
        self.optim_alg = optim_alg
        if optim_alg == 'adam':
            self.algo_optim = AdaM
        elif optim_alg == 'adamax':
            self.algo_optim = AdaMax

        log_f(log_txt, 'VFAE, dim_h_en_z1: ' + str(dim_h_en_z1) + ', dim_h_de_z1: ' + str(dim_h_de_z1) +
              ', dim_h_en_z2: ' + str(dim_h_en_z2) + ', dim_h_de_x: ' + str(dim_h_de_x) + ', dim_h_clf: ' + str(dim_h_clf) +
              ', dim_z1: ' + str(dim_z1) + ', dim_z2: ' + str(dim_z2) + ', nonlinearity: ' + str(nonlinearity) +
              ', type_rec: ' + str(type_rec) + ', dropout: ' + str(dropout_rate) + ', normalization: ' + str(normalization) +
              ', L: ' + str(L) + ', weight_decay: ' + str(self.weight_decay) + ', use_MMD: ' + str(use_MMD) +
              ', kernel_MMD: ' + str(kernel_MMD) + ', mmd_strength: ' + str(lambda_reg) + ', supervised_rate: ' +
              str(supervised_rate) + ', type_y: ' + str(type_y) + ', prior_y: ' + str(prior_y) + ', use_s: ' + str(use_s) +
              ', semi_supervised: ' + str(semi_supervised) + ', optim_alg: ' + optim_alg + ', random_seed: ' +
              str(random_seed))

        self._build_blocks()
        self._create_model()

    def _get_mmd_criterion(self, z, sind):
        mmd = 0
        for j, ind in enumerate(sind):
            z0_ = z[ind]
            aux_z0 = blk.srng.normal((1, z.shape[1]))  # if an S category does not have any samples
            z0 = T.concatenate([z0_, aux_z0], axis=0)
            if len(sind) == 2:
                z1_ = z[sind[j + 1]]
                aux_z1 = blk.srng.normal((1, z.shape[1]))
                z1 = T.concatenate([z1_, aux_z1], axis=0)
                return - blk.mmd_objective(z0, z1, kernel=self.kernel_MMD)
            # z1 = T.concatenate([z, aux_z0], axis=0)
            z1 = z
            mmd += - blk.mmd_objective(z0_, z1, kernel=self.kernel_MMD)

        return mmd

    def copy_params(self, params1, params2, params3):
        params1_inf = [[theano.shared(param.get_value(), name=param.name + '_avg') for param in params] for params in params1]
        params2_inf = [[theano.shared(param.get_value(), name=param.name + '_avg') for param in params] for params in params2]
        params3_inf = [[theano.shared(param.get_value(), name=param.name + '_avg') for param in params] for params in params3]

        return params1_inf, params2_inf, params3_inf

    def _build_blocks(self):
        # create the encoder for z1 q(z1|x,s)
        if self.use_s:
            in_qz1 = [self.dim_x, self.dim_s]
        else:
            in_qz1 = [self.dim_x]
        in_encoder_params_z1 = gpm.create_input_layer(dim_in=in_qz1, dim_h0=self.dim_h_en_z1[0], normed=self.normed,
                                                      appended_name='_en_z1')
        if len(self.dim_h_en_z1) > 1:
            hidden_layers_en_z1 = gpm.append_hidden_layers(dim_in=self.dim_h_en_z1[0], dim_h=self.dim_h_en_z1[1:],
                                                           normed=self.normed, appended_name='_en_z1')
        else:
            hidden_layers_en_z1 = []
        z1_layer = gpm.create_output_layer(dim_in=self.dim_h_en_z1[-1], dim_out=[self.dim_z1, self.dim_z1],
                                           normed=self.normed, appended_name='_en_z1')
        self.encoder_z1 = self.encoder(in_encoder_params_z1, hidden_layers_en_z1, z1_layer, nonlin=self.nonlinearity,
                                       normalization=self.normalization, dropout_rate=self.dropout_rate)
        # for inference copy the parameters to use a moving average
        in_encoder_params_z1_inf, hidden_layers_en_z1_inf, z1_layer_inf = self.copy_params(in_encoder_params_z1,
                                                                                           hidden_layers_en_z1, z1_layer)
        self.encoder_z1_inf = self.encoder(in_encoder_params_z1_inf, hidden_layers_en_z1_inf, z1_layer_inf,
                                           nonlin=self.nonlinearity, normalization=self.normalization,
                                           dropout_rate=self.dropout_rate)

        # create the encoder for z2 q(z2|z1,y)
        in_encoder_params_z2 = gpm.create_input_layer(dim_in=[self.dim_z1, self.dim_y], dim_h0=self.dim_h_en_z2[0],
                                                      normed=self.normed, appended_name='_en_z2')
        if len(self.dim_h_en_z2) > 1:
            hidden_layers_en_z2 = gpm.append_hidden_layers(dim_in=self.dim_h_en_z2[0], dim_h=self.dim_h_en_z2[1:],
                                                           normed=self.normed, appended_name='_en_z2')
        else:
            hidden_layers_en_z2 = []
        z2_layer = gpm.create_output_layer(dim_in=self.dim_h_en_z2[-1], dim_out=[self.dim_z2, self.dim_z2],
                                           normed=self.normed, appended_name='_en_z2')
        self.encoder_z2 = self.encoder(in_encoder_params_z2, hidden_layers_en_z2, z2_layer, nonlin=self.nonlinearity,
                                       normalization=self.normalization, dropout_rate=self.dropout_rate)
        # for inference copy the parameters to use a moving average
        in_encoder_params_z2_inf, hidden_layers_en_z2_inf, z2_layer_inf = self.copy_params(in_encoder_params_z2,
                                                                                           hidden_layers_en_z2, z2_layer)
        self.encoder_z2_inf = self.encoder(in_encoder_params_z2_inf, hidden_layers_en_z2_inf, z2_layer_inf,
                                           nonlin=self.nonlinearity, normalization=self.normalization,
                                           dropout_rate=self.dropout_rate)

        # create the encoder for y q(y|z1)
        if self.dim_h_clf:
            in_clf_params = gpm.create_input_layer(dim_in=[self.dim_z1], dim_h0=self.dim_h_clf[0], normed=self.normed,
                                                   appended_name='_clf')
            if len(self.dim_h_clf) > 1:
                hidden_layers_clf = gpm.append_hidden_layers(dim_in=self.dim_h_clf[0], dim_h=self.dim_h_clf[1:],
                                                             normed=self.normed, appended_name='_clf')
            else:
                hidden_layers_clf = []
        else:
            in_clf_params, hidden_layers_clf = [], []
        in_dim_clf = self.dim_z1 if not self.dim_h_clf else self.dim_h_clf[-1]
        if self.type_y == 'discrete':
            prediction_layer = gpm.create_output_layer(dim_in=in_dim_clf, dim_out=[self.dim_y], normed=self.normed,
                                                       appended_name='_clf')
            self.encoder_y = blk.CategoricalDecoder(in_clf_params, hidden_layers_clf, prediction_layer,
                                                    nonlin=self.nonlinearity, normalization=self.normalization,
                                                    dropout_rate=self.dropout_rate)
            # for inference copy the parameters to use a moving average
            in_clf_params_inf, hidden_layers_clf_inf, prediction_layer_inf = self.copy_params(in_clf_params,
                                                                                              hidden_layers_clf,
                                                                                              prediction_layer)
            self.encoder_y_inf = blk.CategoricalDecoder(in_clf_params_inf, hidden_layers_clf_inf, prediction_layer_inf,
                                                        nonlin=self.nonlinearity, normalization=self.normalization,
                                                        dropout_rate=self.dropout_rate)
        else:  # regression case
            prediction_layer = gpm.create_output_layer(dim_in=in_dim_clf, dim_out=[self.dim_y, self.dim_y],
                                                       normed=self.normed, appended_name='_clf')
            self.encoder_y = blk.DiagGaussianDecoder(in_clf_params, hidden_layers_clf, prediction_layer,
                                                     nonlin=self.nonlinearity, normalization=self.normalization,
                                                     dropout_rate=self.dropout_rate)
            # for inference copy the parameters to use a moving average
            in_clf_params_inf, hidden_layers_clf_inf, prediction_layer_inf = self.copy_params(in_clf_params,
                                                                                              hidden_layers_clf,
                                                                                              prediction_layer)
            self.encoder_y_inf = blk.DiagGaussianDecoder(in_clf_params_inf, hidden_layers_clf_inf, prediction_layer_inf,
                                                         nonlin=self.nonlinearity, normalization=self.normalization,
                                                         dropout_rate=self.dropout_rate)

        # create the decoder for p(z1|z2,y)
        in_decoder_params_z1 = gpm.create_input_layer(dim_in=[self.dim_z2, self.dim_y], dim_h0=self.dim_h_de_z1[0],
                                                      normed=self.normed, appended_name='_de_z1')
        if len(self.dim_h_de_z1) > 1:
            hidden_layers_de_z1 = gpm.append_hidden_layers(dim_in=self.dim_h_de_z1[0], dim_h=self.dim_h_de_z1[1:],
                                                           normed=self.normed, appended_name='_de_z1')
        else:
            hidden_layers_de_z1 = []
        reconstruction_layer_z1 = gpm.create_output_layer(dim_in=self.dim_h_de_z1[-1], dim_out=[self.dim_z1, self.dim_z1],
                                                          normed=self.normed, appended_name='_de_z1')
        self.decoder_z1 = self.decoder_z1_(in_decoder_params_z1, hidden_layers_de_z1, reconstruction_layer_z1,
                                           nonlin=self.nonlinearity, normalization=self.normalization,
                                           dropout_rate=self.dropout_rate)
        in_decoder_params_z1_inf, hidden_layers_de_z1_inf, reconstruction_layer_z1_inf = self.copy_params(in_decoder_params_z1,
                                                                                                          hidden_layers_de_z1,
                                                                                                          reconstruction_layer_z1)
        self.decoder_z1_inf = self.decoder_z1_(in_decoder_params_z1_inf, hidden_layers_de_z1_inf,
                                               reconstruction_layer_z1_inf, nonlin=self.nonlinearity,
                                               normalization=self.normalization, dropout_rate=self.dropout_rate)

        # create the decoder for p(x|z1,s)
        if self.use_s:
            in_px = [self.dim_z1, self.dim_s]
        else:
            in_px = [self.dim_z1]

        in_decoder_params_x = gpm.create_input_layer(dim_in=in_px, dim_h0=self.dim_h_de_x[0], normed=self.normed,
                                                     appended_name='_de_x')
        if len(self.dim_h_de_x) > 1:
            hidden_layers_de_x = gpm.append_hidden_layers(dim_in=self.dim_h_de_x[0], dim_h=self.dim_h_de_x[1:],
                                                          normed=self.normed, appended_name='_de_x')
        else:
            hidden_layers_de_x = []
        if self.type_rec in ['diag_gaussian']:
            reconstruction_layer_x = gpm.create_output_layer(dim_in=self.dim_h_de_x[-1], dim_out=[self.dim_x, self.dim_x],
                                                             normed=self.normed, appended_name='_de_x')
        else:
            reconstruction_layer_x = gpm.create_output_layer(dim_in=self.dim_h_de_x[-1], dim_out=[self.dim_x],
                                                             normed=self.normed, appended_name='_de_x')
        self.decoder_x = self.decoder(in_decoder_params_x, hidden_layers_de_x, reconstruction_layer_x,
                                      nonlin=self.nonlinearity, normalization=self.normalization,
                                      dropout_rate=self.dropout_rate)
        in_decoder_params_x_inf, hidden_layers_de_x_inf, reconstruction_layer_x_inf = self.copy_params(in_decoder_params_x,
                                                                                                       hidden_layers_de_x,
                                                                                                       reconstruction_layer_x)
        self.decoder_x_inf = self.decoder(in_decoder_params_x_inf, hidden_layers_de_x_inf, reconstruction_layer_x_inf,
                                          nonlin=self.nonlinearity, normalization=self.normalization,
                                          dropout_rate=self.dropout_rate)

        # get all the parameters
        params = in_encoder_params_z1 + hidden_layers_en_z1 + z1_layer + in_encoder_params_z2 + hidden_layers_en_z2 + z2_layer + \
            in_decoder_params_z1 + hidden_layers_de_z1 + reconstruction_layer_z1 + in_decoder_params_x + hidden_layers_de_x + reconstruction_layer_x + \
            in_clf_params + hidden_layers_clf + prediction_layer
        params_inf = in_encoder_params_z1_inf + hidden_layers_en_z1_inf + z1_layer_inf + in_encoder_params_z2_inf + hidden_layers_en_z2_inf + z2_layer_inf + \
            in_decoder_params_z1_inf + hidden_layers_de_z1_inf + reconstruction_layer_z1_inf + in_decoder_params_x_inf + hidden_layers_de_x_inf + reconstruction_layer_x_inf + \
            in_clf_params_inf + hidden_layers_clf_inf + prediction_layer_inf
        self.params = [item for sublist in params for item in sublist]
        self.params_inf = [item for sublist in params_inf for item in sublist]

    def _fprop(self, x, z1, zinf1, qz1, qz1_inf, s, y, yinf, encoder_z1, encoder_z1_inf, encoder_z2, encoder_z2_inf,
               decoder_z1, decoder_z1_inf, decoder_x, decoder_x_inf):
        """
        Propagate through the generative model for a given class Y
        """
        # sample from q(z2|z1,y)
        qz2 = encoder_z2.transform([z1, y])
        qz2_inf = encoder_z2_inf.transform([zinf1, yinf], inference=True)
        z2 = encoder_z2.sample(*qz2)
        zinf2 = encoder_z2_inf.sample(*qz2_inf)
        # kl-divergence for z2 q(z2|z1,y) * (logp(z2)/logq(z2|z1,y))
        try:
            kldiv_z2 = encoder_z2.kldivergence_perx(*qz2) / (1. * self.L)
            kldiv_z2_inf = encoder_z2_inf.kldivergence_perx(*qz2_inf) / (1. * self.L)
        except:
            # no KL-divergence, drop to the logp(z) - logq(z) estimator
            kldiv_z2 = (encoder_z2.logp_perx(z2) - encoder_z2.logq_perx(z2, *qz2)) / (1. * self.L)
            kldiv_z2_inf = (encoder_z2_inf.logp_perx(zinf2) - encoder_z2.logq_perx(zinf2, *qz2_inf)) / (1. * self.L)

        # p(z1|z2,y)
        pz1 = decoder_z1.transform([z2, y])
        pz1_inf = decoder_z1_inf.transform([zinf2, yinf], inference=True)
        # kl-divergence for z1 q(z1|x,s) * (logp(z1|y,z2)/logq(z1|x,s))
        kldiv_z1 = (decoder_z1.logp_perx(z1, *pz1) - encoder_z1.logq_perx(z1, *qz1)) / (1. * self.L)
        kldiv_z1_inf = (decoder_z1_inf.logp_perx(zinf1, *pz1_inf) - encoder_z1.logq_perx(zinf1, *qz1_inf)) / (1. * self.L)
        # p(x|z1,s)
        if self.use_s:
            if self.type_rec == 'diag_gaussian':
                px = decoder_x.transform([z1, s], constrain_means=self.constrain_means)
                px_inf = decoder_x_inf.transform([zinf1, s], constrain_means=self.constrain_means, inference=True)
            else:
                px = decoder_x.transform([z1, s])
                px_inf = decoder_x_inf.transform([zinf1, s], inference=True)
        else:
            if self.type_rec == 'diag_gaussian':
                px = decoder_x.transform([z1], constrain_means=self.constrain_means)
                px_inf = decoder_x_inf.transform([zinf1], constrain_means=self.constrain_means, inference=True)
            else:
                px = decoder_x.transform([z1])
                px_inf = decoder_x_inf.transform([zinf1], inference=True)
        # get the reconstruction loss
        logpx = decoder_x.logp_perx(x, *px) / (1. * self.L)
        logpx_inf = decoder_x_inf.logp_perx(x, *px_inf) / (1. * self.L)

        bound = kldiv_z2 + kldiv_z1 + logpx
        boundinf = kldiv_z2_inf + kldiv_z1_inf + logpx_inf
        return bound, boundinf

    def _create_model(self):
        # for the encoder
        x = T.matrix('x')
        x_u = T.matrix('x_u')
        s_ = T.lvector('s')
        s_u_ = T.lvector('s_u')
        if self.type_y == 'discrete':
            y = T.lvector('y')
            if self.prior_y == 'uniform':
                pr_y_l = T.ones((x.shape[0], self.dim_y)) / (1. * self.dim_y)
                pr_y_u = T.ones((x_u.shape[0], self.dim_y)) / (1. * self.dim_y)
            else:
                pr_y_l = T.ones((x.shape[0], self.dim_y)) * self.prior_y
                pr_y_u = T.ones((x_u.shape[0], self.dim_y)) * self.prior_y
            y1ofk = T.as_tensor_variable(T.extra_ops.to_one_hot(y, self.dim_y))
            y1ofk.name = 'nuisance_y'
        else:
            y = T.matrix('y')
            y1ofk = T.as_tensor_variable(y)
            y1ofk.name = 'nuisance_y'

        if self.semi_supervised:
            S = T.concatenate([s_, s_u_], axis=0)
        else:
            S = s_
        # indices for labeled data
        xind = T.arange(0, x.shape[0])
        # indices for unlabeled data
        if self.semi_supervised:
            xind_u = T.arange(x.shape[0], x.shape[0] + x_u.shape[0])

        if self.use_MMD:
            # get the indices for the nuisance variable groups
            sind = []
            for si in range(self.dim_s):
                sind.append(T.eq(S, si).nonzero()[0])

        S2 = T.extra_ops.to_one_hot(S, self.dim_s)
        s_l = T.as_tensor_variable(S2[xind])
        s_l.name = 'nuisance_s'
        if self.semi_supervised:
            s_u = T.as_tensor_variable(S2[xind_u])
            s_u.name = 'nuisance_s'

        # construct the objective function
        if self.use_s:
            qz1_l = self.encoder_z1.transform([x, s_l])
            qz1_inf_l = self.encoder_z1_inf.transform([x, s_l], inference=True)
            if self.semi_supervised:
                qz1_u = self.encoder_z1.transform([x_u, s_u])
                qz1_inf_u = self.encoder_z1_inf.transform([x_u, s_u], inference=True)
        else:
            qz1_l = self.encoder_z1.transform([x])
            qz1_inf_l = self.encoder_z1_inf.transform([x], inference=True)
            if self.semi_supervised:
                qz1_u = self.encoder_z1.transform([x_u])
                qz1_inf_u = self.encoder_z1_inf.transform([x_u], inference=True)

        mmd, mmd_inf = T.as_tensor_variable(0.), T.as_tensor_variable(0.)
        bound_l, bound_u = 0, T.as_tensor_variable(0.)
        bound_l_inf, bound_u_inf = 0, T.as_tensor_variable(0.)
        ly, ly_inf = 0, 0

        for i in range(self.L):
            # objective for labeled data
            # sample from q(z1|x, s)
            z1_l = self.encoder_z1.sample(*qz1_l)
            zinf_1_l = self.encoder_z1_inf.sample(*qz1_inf_l)
            bounds, bounds_inf = self._fprop(x, z1_l, zinf_1_l, qz1_l, qz1_inf_l, s_l, y1ofk, y1ofk, self.encoder_z1,
                                             self.encoder_z1_inf, self.encoder_z2, self.encoder_z2_inf, self.decoder_z1,
                                             self.decoder_z1_inf, self.decoder_x, self.decoder_x_inf)

            bound_l += T.sum(bounds)  # the logprior of y is ommited as it is constant wrt the optimization
            bound_l_inf += T.sum(bounds_inf)
            # get the classification or regression loss for those labels that are observed
            alpha = self.supervised_rate * (x.shape[0] + x_u.shape[0]) / (1. * x.shape[0])
            py_l = self.encoder_y.transform([z1_l])
            pyinf_l = self.encoder_y_inf.transform([zinf_1_l], inference=True)

            ly += alpha * self.encoder_y.logp(y, *py_l) / (1. * self.L)
            ly_inf += alpha * self.encoder_y_inf.logp(y, *pyinf_l) / (1. * self.L)

            # objective for unlabeled data
            # sample from q(z1|x, s) for the unlabeled
            if self.semi_supervised:
                z1_u = self.encoder_z1.sample(*qz1_u)
                zinf_1_u = self.encoder_z1_inf.sample(*qz1_inf_u)
                # get q(y|z1)
                py_u = self.encoder_y.transform([z1_u])
                pyinf_u = self.encoder_y_inf.transform([zinf_1_u], inference=True)

                if self.type_y == 'discrete':
                    # marginalize out y
                    for j in range(self.dim_y):
                        y_j = T.extra_ops.to_one_hot(T.ones((x_u.shape[0],), dtype=int) * j, self.dim_y)
                        bounds_, bounds_inf_ = self._fprop(x_u, z1_u, zinf_1_u, qz1_u, qz1_inf_u, s_u, y_j, y_j,
                                                           self.encoder_z1, self.encoder_z1_inf, self.encoder_z2,
                                                           self.encoder_z2_inf, self.decoder_z1, self.decoder_z1_inf,
                                                           self.decoder_x, self.decoder_x_inf)
                        bound_u += T.sum(py_u[0][:, j] * bounds_)
                        bound_u_inf += T.sum(pyinf_u[0][:, j] * bounds_inf_)
                    bound_u += self.encoder_y.kldivergence(pr_y_u, py_u[0]) / (1. * self.L)
                    bound_u_inf += self.encoder_y_inf.kldivergence(pr_y_u, pyinf_u[0]) / (1. * self.L)
                else:
                    # if continous then just use SGVB and sample y
                    y_j = self.encoder_y.sample(*py_u)
                    y_jinf = self.encoder_y_inf.sample(*pyinf_u)
                    bounds_, bounds_inf_ = self._fprop(x_u, z1_u, zinf_1_u, qz1_u, qz1_inf_u, s_u, y_j, y_jinf,
                                                       self.encoder_z1, self.encoder_z1_inf, self.encoder_z2,
                                                       self.encoder_z2_inf, self.decoder_z1, self.decoder_z1_inf,
                                                       self.decoder_x, self.decoder_x_inf)
                    bound_u += T.sum(bounds_) + (self.encoder_y.kldivergence(*(py_u + self.prior_y)) / (1. * self.L))
                    bound_u_inf += T.sum(bounds_inf_) + (self.encoder_y_inf.kldivergence(*(pyinf_u + self.prior_y)) / (1. * self.L))

            # maximum mean discrepancy regularization
            if self.use_MMD:
                if self.semi_supervised:
                    Z = T.concatenate([z1_l, z1_u], axis=0)
                    Zinf = T.concatenate([zinf_1_l, zinf_1_u], axis=0)
                else:
                    Z = z1_l
                    Zinf = zinf_1_l
                beta = self.lambda_reg * (Z.shape[0])
                betainf = self.lambda_reg * (Zinf.shape[0])
                mmd += beta * self._get_mmd_criterion(Z, sind) / (1. * self.L)
                mmd_inf += betainf * self._get_mmd_criterion(Zinf, sind) / (1. * self.L)

        # for the new representation average samples over the invariant posterior
        M = 1  # how many samples
        trans_z_l = T.zeros((x.shape[0], self.dim_z1))
        if self.semi_supervised:
            trans_z_u = T.zeros((x_u.shape[0], self.dim_z1))
        for j in range(M):
            trans_z_l += self.encoder_z1_inf.sample(*qz1_inf_l) / (1. * M)
            if self.semi_supervised:
                trans_z_u += self.encoder_z1_inf.sample(*qz1_inf_u) / (1. * M)
        if self.semi_supervised:
            trans_z = T.concatenate([trans_z_l, trans_z_u], axis=0)
        else:
            trans_z = trans_z_l

        # for predicting
        pypred = self.encoder_y_inf.transform([trans_z], inference=True)
        if self.type_y == 'discrete':
            classifier_pred = self.encoder_y_inf.most_probable(*pypred)
            proba = pypred[0]
        else:
            classifier_pred, proba = pypred

        # stack the objectives
        objectives = [bound_l, bound_u, ly, mmd]
        # stack the evaluation objectives
        objectives_inference = [bound_l_inf, bound_u_inf, ly_inf, mmd_inf]

        self.predict = theano.function([x, x_u, s_, s_u_], [classifier_pred, proba], on_unused_input='ignore')
        self.transform = theano.function([x, x_u, s_, s_u_], trans_z, on_unused_input='ignore')

        self.optimizer = self.algo_optim(objectives, objectives_inference, [x, x_u, s_, s_u_, y], self.params, self.params_inf,
                                         mode=self.mode, alpha=self.learningRate, regularization=self.regularization,
                                         normalization=self.normalization, weight_decay=self.weight_decay,
                                         batch_size=self.batch_size, replace_empty=False, beta1=self.beta1, beta2=self.beta2,
                                         polyak=self.polyak, beta3=self.beta3)

    def fit(self, x, s, y, x_u=None, s_u=None, xvalid=None, xvalid_u=None, svalid=None, svalid_u=None, yvalid=None,
            verbose=False, print_every=10):
        num_u, num_valid_u = 1, 1
        if s_u is not None:
            indices_u = np.arange(x_u.shape[0])
            gpm.prng.shuffle(indices_u)
            num_u = x_u.shape[0]
        if svalid_u is not None:
            num_valid_u = xvalid_u.shape[0]

        indices_t = np.arange(x.shape[0])
        indices_v = np.arange(xvalid.shape[0])
        gpm.prng.shuffle(indices_t)
        gpm.prng.shuffle(indices_v)
        rounding = lambda numbers: ['%.3f' % num for num in numbers]
        dummy_xu, dummy_su = np.zeros((0, x.shape[1]), dtype=theano.config.floatX), np.zeros((0,), dtype=np.int32)

        stats_train, stats_valid = [], []
        t = time.time()
        for i in range(self.iterations + 1):
            if x_u is not None:
                xtrain, xtrain_u = x[indices_t], x_u[indices_u]
                strain, strain_u = s[indices_t], s_u[indices_u]
                ytrain = y[indices_t]
            else:
                xtrain, xtrain_u = x[indices_t], dummy_xu
                strain, strain_u = s[indices_t], dummy_su
                ytrain = y[indices_t]
            # training
            objectives = self.optimizer.train([xtrain, xtrain_u, strain, strain_u, ytrain], verbose=verbose).tolist()
            objectives[0] /= xtrain.shape[0]
            objectives[1] /= num_u
            objectives[2] /= (xtrain.shape[0] + num_u)
            objectives[3] /= (x.shape[0] + num_u)  # n_batches

            # validation
            if xvalid is not None:
                if xvalid_u is None:
                    xvalid_u, svalid_u = dummy_xu, dummy_su

                objectives_v = self.optimizer.evaluate([xvalid, xvalid_u, svalid, svalid_u, yvalid]).tolist()
                objectives_v[0] /= xvalid.shape[0]
                objectives_v[1] /= num_valid_u
                objectives_v[2] /= (xvalid.shape[0] + num_valid_u)
                objectives_v[3] /= (xvalid.shape[0] + num_valid_u)

            gpm.prng.shuffle(indices_t); gpm.prng.shuffle(indices_v)
            if s_u is not None:
                gpm.prng.shuffle(indices_u)

            summed_obj = np.sum(objectives)
            ypred, p = self.predict(x, dummy_xu, s, dummy_su)
            if self.type_y == 'discrete':
                t_e = (ypred == y).sum() / (1. * y.shape[0])
                t_err = [summed_obj, t_e]
            else:
                t_e = np.sqrt(np.mean((ypred - y) ** 2))
                mt_e = np.mean(np.abs(ypred - y))
                t_err = [summed_obj, t_e, mt_e]
            objectives.extend(t_err)
            stats_train.append(objectives)

            if xvalid is not None:
                summed_obj_v = np.sum(objectives_v)
                ypred_v, _ = self.predict(xvalid, dummy_xu, svalid, dummy_su)
                if self.type_y == 'discrete':
                    v_e = (ypred_v == yvalid).sum() / (1. * yvalid.shape[0])
                    v_err = [summed_obj_v, v_e]
                else:
                    v_e = np.sqrt(np.mean((ypred_v - yvalid) ** 2))
                    mv_e = np.mean(np.abs(ypred_v - yvalid))
                    v_err = [summed_obj_v, v_e, mv_e]
                objectives_v.extend(v_err)
                stats_valid.append(objectives_v)
            if i % print_every == 0:
                dt = time.time() - t
                t = time.time()
                string = 'Epoch: ' + str(i) + '/' + str(self.iterations) + ' train: ' +  str(map(float,rounding(objectives)))
                if xvalid is not None:
                    string += ' valid: ' + str(map(float, rounding(objectives_v)))
                string += ' time: ' + '%.3f' % dt
                log_f(self.log_txt, string)

        return stats_train, stats_valid
