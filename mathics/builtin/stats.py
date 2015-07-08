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

class SympyDistribution(SympyFunction):
    pass

class NormalDistribution(SympyDistribution):
    sympy_name = 'NormalDistribution'

    def to_sympy(self, expr):
        a, b = expr.leaves[0], expr.leaves[1]
        return stats.Normal(str(uuid.uuid4()), a.to_sympy(), b.to_sympy(positive=True))

    def apply(self, mu, sigma, evaluation):
        'NormalDistribution[mu_, sigma_]'
        return None 


class WeibullDistribution(SympyDistribution):
    sympy_name = 'WeibullDistribution'

    def to_sympy(self, expr):
        a, b = expr.leaves[0], expr.leaves[1]
        return stats.Weibull(str(uuid.uuid4()), a.to_sympy(), b.to_sympy())

    def apply(self, alpha, beta, evaluation):
        'WeibullDistribution[alpha_, beta_]'
        return None 


class ChiDistribution(SympyDistribution):
    sympy_name = 'ChiDistribution'

    def to_sympy(self, expr):
        k = expr.leaves[0]
        return stats.Chi(str(uuid.uuid4()), k.to_sympy())

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
