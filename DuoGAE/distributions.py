import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions import Distribution
from pyro.distributions.bernoulli import Bernoulli
from pyro.distributions.random_primitive import RandomPrimitive
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss


class WeightedBernoulli(Bernoulli):
    """Bernoulli distribution with a weighted cross entropy. Used for imbalanced data when you 
    want to increase the penalizization of the positive class. """

    def __init__(self, *args, **kwargs):
        self.weight = kwargs.pop('weight', 1.0)
        super(WeightedBernoulli, self).__init__(*args, **kwargs)

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        max_val = (-self.logits).clamp(min=0)
        # ----------
        # The following two lines are the only difference between WeightedBernoulli and Bernoulli
        # Cf. derivation at:
        # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
        weight = (1 + (self.weight - 1) * x)
        binary_cross_entropy = self.logits * (1 - x) + max_val + weight * ((-max_val).exp() + (-self.logits - max_val).exp()).log()
        # ----------
        log_prob = -binary_cross_entropy
        # XXX this allows for the user to mask out certain parts of the score, for example
        # when the data is a ragged tensor. also useful for KL annealing. this entire logic
        # will likely be done in a better/cleaner way in the future
        if self.log_pdf_mask is not None:
            log_prob = log_prob * self.log_pdf_mask
        return torch.sum(log_prob, -1).contiguous().view(batch_log_pdf_shape)


class VonMisesFisher(Distribution):
    """
    Hyperspherical von Mises - Fisher distribution.

    The probability density function (pdf) is:
   
    pdf(x; mu, k) = exp(k mu^T x) / Z
    Z = (k ** (m / 2 - 1)) / ((2pi ** m / 2) * besseli(m / 2 - 1, k))

    where loc = mu is the mean, scale = k is the concentration, 
    m is the dimensionality, and Z is the normalization constant.
    
    See https://en.wikipedia.org/wiki/Von_Mises-Fisher distribution for more details on the 
    Von Mises-Fiser distribution.

    """
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        super(VonMisesFisher, self).__init__(reparametrized=True)


    def batch_shape(self, x=None):
        """
        The left-hand tensor shape of samples, used for batching.
        Samples are of shape d.shape(x) == d.batch_shape(x) + d.event_shape()
        """
        pass


    def event_shape(self, x=None):
        """
        The right-hand tensor shape of samples, used for individual events.
        The event dimension(/s) is used to designate random variables that
        could potentially depend on each other, for instance in the case of
        Dirichlet or the categorical distribution, but could also simply
        be used for logical grouping, for example in the case of a normal distribution
        with a diagonal covariance matrix.

        Samples are of shape d.shape(x) == d.batch_shape(x) + d.event_shape().
        """
        pass


    def sample(self):
        pass


    def batch_log_pdf(self, x):
        """
        Evaluates log probability densities for each of a batch of samples.
        """
        pass


    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return self.loc


    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        pass


# function aliases
weighted_bernoulli = RandomPrimitive(WeightedBernoulli)


