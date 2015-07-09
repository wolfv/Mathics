# -*- coding: utf8 -*-

"""
Statistics
"""

import sympy
import uuid
import sympy.stats as stats

from mathics.builtin.base import Builtin, SympyFunction
from mathics.core.convert import from_sympy
from mathics.core.expression import Expression, Integer

varcounter = 0

def internal_variable_name():
    global varcounter
    varcounter += 1
    return '__rv_internal_' + str(varcounter)

class SympyDistribution(SympyFunction):
    pass

class NormalDistribution(SympyDistribution):
    sympy_name = 'NormalDistribution'

    def to_sympy(self, expr):
        a, b = expr.leaves[0], expr.leaves[1]
        return stats.Normal(internal_variable_name(), a.to_sympy(), b.to_sympy(positive=True))

    def apply(self, mu, sigma, evaluation):
        'NormalDistribution[mu_, sigma_]'
        return None 


class WeibullDistribution(SympyDistribution):
    sympy_name = 'WeibullDistribution'

    def to_sympy(self, expr):
        a, b = expr.leaves[0], expr.leaves[1]
        return stats.Weibull(internal_variable_name(), a.to_sympy(), b.to_sympy())

    def apply(self, alpha, beta, evaluation):
        'WeibullDistribution[alpha_, beta_]'
        return None 



# class DiscreteUniformDistribution(SympyDistribution):
#     sympy_name = 'DiscreteUniform'

#     def to_sympy(self, expr):
#         a = expr.leaves[0]
#         return stats.DiscreteUniform(internal_variable_name(), a.to_sympy())

#     def apply(self, alpha, beta, evaluation):
#         'BernoulliDistribution[alpha_, beta_]'
#         return None 


class BernoulliDistribution(SympyDistribution):
    sympy_name = 'BernoulliDistribution'

    def to_sympy(self, expr):
        p = expr.leaves[0]
        return stats.Bernoulli(internal_variable_name(), p.to_sympy())

    def apply(self, p, evaluation):
        'BernoulliDistribution[p_]'
        return None 



class ChiDistribution(SympyDistribution):
    sympy_name = 'ChiDistribution'

    def to_sympy(self, expr):
        k = expr.leaves[0]
        return stats.Chi(internal_variable_name(), k.to_sympy())

    def apply(self, k, evaluation):
        'ChiDistribution[k_]'
        return None 

class PDF(Builtin):

    def apply(self, distribution, n, evaluation):
        'PDF[distribution_, n_]'
        n = n.to_sympy()
        return from_sympy(stats.density(distribution.to_sympy())(n))


class CDF(Builtin):
    def apply(self, distribution, n, evaluation):
        'CDF[distribution_, n_]'
        n = n.to_sympy()
        return from_sympy(stats.cdf(distribution.to_sympy())(n))


class Mean(Builtin):
    """
    <dl>
    <dt>'Mean[$list$]'
    <dd>
        Mean over all values in list (Total[list] / Length[list])
    </dd>
    </dt>
    </dl>
    >> Mean[{1,2,3}]
     = 2
    >> Mean[{a, b, 213}]
     = (123 + a + b) / 3
    """

    rules = {
        'Mean[list_List]': 'Total[list] / Length[list]'
    }

    messages = {
        'onlylistorrandom': 'Mean can only be taken from lists or distributions'
    }

    def apply(self, dist, evaluation):
        'Mean[dist_]'

        dist = dist.to_sympy()
        if not type(dist) == stats.rv.RandomSymbol:
            evaluation.message('Mean', 'onlylistorrandom')
        else:
            return from_sympy(stats.E(dist))

class Variance(Builtin):
    """
    <dl>
    <dt>'Variance[$list$]'
    <dd>
        Mean over all values in list (Total[list] / Length[list])
    </dd>
    </dt>
    </dl>
    >> Mean[{1,2,3}]
     = 2
    >> Mean[{a, b, 213}]
     = (123 + a + b) / 3
    """

    rules = {
        'Variance[list_List]': '(list-Mean[list]).Conjugate[list-Mean[list]]/(Length[list]-1)'
    }

    messages = {
        'onlylistorrandom': 'Mean can only be taken from lists or distributions'
    }

    def apply(self, dist, evaluation):
        'Variance[dist_]'

        dist = dist.to_sympy()
        if not type(dist) == stats.rv.RandomSymbol:
            evaluation.message('Variance', 'onlylistorrandom')
        else:
            return from_sympy(stats.variance(dist))


class Covariance(Builtin):

    rules = {
        'Covariance[vec1_List, vec2_List]': '(vec1 - Mean[vec1]).Conjugate[vec2 - Mean[vec2]] / (Length[vec1]-1)'
    }

class RandomVariate(Builtin):
    # TODO:
    # RandomVariate[dist, {n1, n2, ... }] -> array of n1 x n2 ... 
    rules = {
        'RandomVariate[dist_]': 'RandomVariate[dist, 1]'
    }

    def apply(self, dist, samples, evaluation):
        'RandomVariate[dist_, samples_Integer]'

        dist = dist.to_sympy()
        samples = samples.to_sympy()
        if not type(dist) == stats.rv.RandomSymbol:
            evaluation.message('Variance', 'onlylistorrandom')
        elif samples == 1:
            return from_sympy(stats.sample(dist))
        elif samples > 1:
            iterator = stats.sample_iter(dist, numsamples=samples)
            return Expression('List', *list(iterator))

class Probability(Builtin):
    pass
    # TODO
    # Probability[x >= 3, x \[Distributed] NormalDistribution[1, 10]]
    # = ...

    # def apply():
