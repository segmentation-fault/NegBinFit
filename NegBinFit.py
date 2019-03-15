__author__ = 'antonio franco'

'''
Copyright (C) 2018  Antonio Franco (antonio_franco@live.it)
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from copy import deepcopy
from scipy.special import gammaln
import numpy as np
from scipy.integrate import cumtrapz
from scipy import optimize
from random import random


class NegBinFit(object):
    """This class fits two negative distributions with mean m1 and m2 respectively, and variance (m1 + m1^2/r) and (m2 +
       m2^2/r2) respectively, using the Frank copula with parameter theta, to the PMF P.

        Args:
            P (two dimensional list): target Probability Mass Function (PMF)
            x1 (int, int): x range from which the PMF starts and ends
            x2 (int, int): y range from which the PMF starts and ends

        Attributes:
            m1, m2 (float): means of the two fitted distributions.
            r1, r2 (float): second parameters of the two fitted distributions.
            RMSE (float): Root Mean Square Error of the fit.
            jCDF (2d ndarray): estimate joint CDF

        """

    def __init__(self, P, x1, x2):
        self.x1 = np.arange(x1[0], x1[1] + 1)
        self.x2 = np.arange(x2[0], x2[1] + 1)
        self.P = deepcopy(np.asarray(P))
        self.targetCDF = self._get_CDF_from_PMF(self.P)

        self.m1 = 0
        self.m2 = 0

        self.r1 = 0
        self.r2 = 0

        self._CDF1 = []
        self._CDF2 = []

        self.jCDF = np.zeros(self.P.shape)

        self.RMSE = 1e3

        self.msg = ""

    def _get_CDF_from_PMF(self, f):
        """Calculates the bivariate CDF from the bivariate PMF f

                Args:
                    f (bidimensional ndarray): PMF.

                Returns:
                    CDF.

                """
        F = np.zeros(f.shape)

        for i in np.arange(0, f.shape[0]):
            for j in np.arange(0, f.shape[1]):
                V = f[0:i, 0:j].flatten()
                F[i][j] = np.sum(V)

        return F

    def _get_PMF_from_CDF(self, F):
        """Calculates the bivariate PMF from the bivariate CDF F

                Args:
                    F (bidimensional ndarray): CDF.

                Returns:
                    PMF.

                """
        f = np.diff(np.diff(F, axis=1), axis=0)

        f = np.c_[np.zeros(f.shape[0]), f]
        f = np.vstack((f, np.zeros((1, f.shape[1]))))

        return f

    def _frank_copula(self, u, v, theta):
        """Calculates the Frank Copula with parameter theta in u and v

                Args:
                    u (float): x value where to calculate Frank Copula = CDF1[i].
                    v (float): y value where to calculate Frank Copula = CDF2[j].
                    theta (float): parameter of the Frank Copula.

                Returns:
                    Frank Copula in u,v.

                """
        F = 1e3
        if np.abs(theta) > 1e-3:
            F = -1.0 / theta * np.log(
                1.0 + (np.exp(- theta * u) - 1.0) * (np.exp(- theta * v) - 1.0) / (np.exp(-theta) - 1.0))

        return F

    def _get_copula(self, i, j, theta):
        """Calculates the Frank Copula with parameter theta in i and j

                Args:
                    i (integer): where to calculate the x CDF to fit.
                    j (integer): where to calculate the y CDF to fit.
                    theta (float): parameter of the Frank Copula.

                Returns:
                    Frank Copula in i, j.

                """
        u = self._CDF1[np.nonzero(self.x1 == i)]
        v = self._CDF2[np.nonzero(self.x2 == j)]
        F = self._frank_copula(u, v, theta)

        return F

    def _neg_bin_PDF(self, r, m, k):
        """Calculates the PMF of the negative binomial distribution with mean m and variance m + m^2/r in k

                Args:
                    m (float): means of the distribution.
                    r (float): second parameter of the distribution.
                    k (integer): value in which to calculate the distribution.

                Returns:
                    PMF of the negative binomial distribution in k.

                """
        if r <= 0 or m < 0 or k < 0:
            return 1e3

        p = float(r / (r + m))
        q = 1.0 - p

        f = np.exp(gammaln(r + k) - gammaln(r) - gammaln(k + 1)) * p ** r * q ** k

        return f

    def _build_CDF(self, r, m, my_range):
        """Calculates the CDF of the negative binomial distribution with mean m and variance m + m^2/r from 0 to N

                Args:
                    m (float): means of the distribution.
                    r (float): second parameter of the distribution.
                    my_range (list of integers): range in which to calculate the CDF.

                Returns:
                    CDF of the negative binomial distribution until N.

                """
        X = my_range
        PDF = list(map(lambda x: self._neg_bin_PDF(r, m, x), X))
        CDF = cumtrapz(PDF, X, initial=0)
        return CDF

    def _build_CDF2(self, r1, r2, m1, m2):
        """Builds the CDFs of the two negative binomial distributions with mean mi and variance mi + mi^2/ri in k

                Args:
                    m1, m2 (float): means of the two fitted distributions.
                    r1, r2 (float): second parameters of the two fitted distributions.
                    theta (float): parameter of the Frank Copula.

                Returns:
                    Nothing

                """
        self._CDF1 = []
        self._CDF2 = []

        self._CDF1 = self._build_CDF(r1, m1, self.x1)
        self._CDF2 = self._build_CDF(r2, m2, self.x2)

    def _build_jCDF(self, r1, r2, m1, m2, theta):
        """Builds the joint CDF of the two negative binomial distributions with mean mi and variance mi + mi^2/ri in k,
            and coefficient theta for the Frank copula

                Args:
                    m1, m2 (float): means of the two fitted distributions.
                    r1, r2 (float): second parameters of the two fitted distributions.

                Returns:
                    Nothing

                """
        self._build_CDF2(r1, r2, m1, m2)

        self.jCDF = np.zeros(self.P.shape)
        for i in range(0, self.P.shape[0]):
            for j in range(0, self.P.shape[1]):
                self.jCDF[i][j] = self._get_copula(self.x1[i], self.x2[j], theta)

    def _get_RMSE(self, A, B):
        """Calculates the RMSE of estimate A vs target B

                Args:
                    A (array of floats): estimate.
                    B (array of floats): target.

                Returns:
                    RMSE.

                """
        assert (len(A) == len(B))

        err = np.sqrt(np.mean((A - B) ** 2))

        return err

    def _err_fun(self, X):
        """Calculates the RMSE of estimated joint CDF vs the target joint CDF

                Args:
                    X (list of floats): [r1, r2, m1, m2, theta]

                Returns:
                    RMSE.

                """
        r1 = X[0]
        r2 = X[1]

        m1 = X[2]
        m2 = X[3]

        theta = X[4]

        self._build_jCDF(r1, r2, m1, m2, theta)

        target = np.asarray(self.targetCDF.flatten())
        estim = np.asarray(self.jCDF.flatten())

        f = self._get_RMSE(estim, target)

        return f

    def go_fit(self):
        """Starts the fitting process
                """

        bnds = ((1e-6, np.inf), (1e-6, np.inf), (0, np.inf), (0, np.inf), (-np.inf, np.inf))

        X0 = [random(), random(), random(), random(), random()]

        X = optimize.minimize(self._err_fun, np.asarray(X0), bounds=bnds)

        self.r1 = X.x[0]
        self.r2 = X.x[1]
        self.m1 = X.x[2]
        self.m2 = X.x[3]
        self.theta = X.x[4]

        self.RMSE = X.fun

        self.msg = X

        self._build_jCDF(self.r1, self.r2, self.m1, self.m2, self.theta)

    def __str__(self):
        n_char_sep = 100

        pstr = "(r1, m1) =  (" + str(self.r1) + ", " + str(self.m1) + ")\n"
        pstr += "(r2, m2) =  (" + str(self.r2) + ", " + str(self.m2) + ")\n"
        pstr += "theta = " + str(self.theta) + "\n"
        pstr += "RMSE = " + str(self.RMSE) + "\n"
        for i in range(0, n_char_sep):
            pstr += "#"
        pstr += "\n"
        pstr += str(self.msg)
        pstr += "\n"
        for i in range(0, n_char_sep):
            pstr += "#"
        pstr += "\n"

        return pstr

    def get_estimate_PMF(self):
        """Calculates the estimated PMF

                Returns:
                    PMF.

                """
        self._build_jCDF(self.r1, self.r2, self.m1, self.m2, self.theta)

        f = self._get_PMF_from_CDF(self.jCDF)

        return f


def negative_multinomial_PDF(k0, p1, p2, x1, x2):
    """Calculates the PMF of the bivariate negative binomial distribution with mean m and variance m + m^2/r in k

            Args:
                k0 (int): the number of failures before the experiment is stopped.
                p1, p2 (float): vector of ssuccess probabilities. p1 + p2 <= 1.
                x1, x2 (integer): values in which to calculate the distribution.

            Returns:
                PMF of the bivariate negative binomial distribution in k.

            """
    assert (p1 + p2 <= 1)
    p0 = 1.0 - p1 - p2
    f = np.exp(gammaln(k0 + x1 + x2) - gammaln(k0) - gammaln(x1 + 1) - gammaln(x2 + 1)) * p0 ** k0 * p1 ** x1 * p2 ** x2
    return f


# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Example with a bivariate negative multinomial
    n_samples = 50
    x = np.arange(0, n_samples)
    y = np.arange(0, n_samples)
    p1 = 0.6
    p2 = 0.2
    k0 = 5

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(X.shape)

    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            Z[i, j] = negative_multinomial_PDF(k0, p1, p2, X[i, j], Y[i, j])

    NFit = NegBinFit(Z, (np.min(x), np.max(x)), (np.min(y), np.max(y)))

    NFit.go_fit()

    Zest = np.asarray(NFit.get_estimate_PMF())

    print(NFit)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c='b', marker='o', label = 'target')
    ax.scatter(X.flatten(), Y.flatten(), Zest.flatten(), c='r', marker='x', label = 'fit')
    plt.title('PMF')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X.flatten(), Y.flatten(), NFit.targetCDF.flatten(), c='b', marker='o', label = 'target')
    ax.scatter(X.flatten(), Y.flatten(), NFit.jCDF.flatten(), c='r', marker='x', label = 'fit')
    plt.title('CDF')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()

    plt.show()
