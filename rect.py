from __future__ import annotations
from tabulate import tabulate

__author__ = 'Drayrs'
__created__ = 240422

import pprint
import warnings
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')
pp = pprint.PrettyPrinter()


class Spreader:

    def __init__(
            self,
            l: float,
            w: npt.NDArray,
            l1: npt.NDArray,
            w1: npt.NDArray,
            t: npt.NDArray,
            h: npt.NDArray,
            k: float = 390,
            M: int = 40,
            N: int = 40
        ):
        self.l = l
        self.k = k
        self.M = M
        self.N = N
        self.m, self.n, self._w, self._l1, self._w1, self._t, self._h = np.meshgrid(
            np.arange(1, self.M + 1),
            np.arange(1, self.N + 1),
            w,
            l1,
            w1,
            t,
            h,
            indexing='ij'
        )
        self._beta = self._w / self.l
        self._epsilon = self._l1 / self.l
        self._gamma = self._w1 / self._w
        self._tau = self._t / self.l
        self._Bi = self.h * self._t / self.k
        self.mpi = self.m * np.pi
        self.npi = self.n * np.pi
        self.A_b = 4 * self.l * self.w
        self.A_s = 4 * self.l1 * self.w1
        self.R_m = self.t / self.k / self.A_b
        self.R_f = 1 / self.h / self.A_b


    @property
    def w(self): return self._w[0, 0, :]

    @property
    def l1(self): return self._l1[0, 0, :]

    @property
    def w1(self): return self._w1[0, 0, :]

    @property
    def t(self): return self._t[0, 0, :]

    @property
    def h(self): return self._h[0, 0, :]

    @property
    def beta(self): return self._beta[0, 0, :]

    @property
    def epsilon(self): return self._epsilon[0, 0, :]

    @property
    def gamma(self): return self._gamma[0, 0, :]

    @property
    def tau(self): return self._tau[0, 0, :]

    @property
    def Bi(self): return self._Bi[0, 0, :]

    def __repr__(self) -> str:
        repr = {
            'w\n[m]': self.w,
            'l1\n[m]': self.l1,
            'w1\n[m]': self.w1,
            't\n[m]': self.t,
            'h\n[W m-2 K-1]': self.h,
            'Bi\n': self.Bi,
            'psi\n': self.psi,
            'R_s\n[K W-1]': self.R_s,
            'R_m\n[K W-1]': self.R_m,
            'R_f\n[K W-1]': self.R_f,
            'R_th\n[K W-1]': self.R_th
        }
        try:
            return tabulate(repr, headers='keys')
        except:
            return pp.pformat(repr)


class FengSpreader(Spreader):

    def C_m0(self):
        a = self.tau * self.mpi
        b = (a + self.Bi)*np.exp(a)
        c = (a - self.Bi)*np.exp(-1 * a)
        return self.gamma * np.sin(self.epsilon*self.mpi) * (b + c) / (b - c) / np.square(self.mpi)
    

    def C_0n(self):
        a = self.tau * self.npi
        b = (a + self.beta*self.Bi)*np.exp(a / self.beta)
        c = (a - self.beta*self.Bi)*np.exp(-1 * a / self.beta)
        return self.beta * self.epsilon * np.sin(self.gamma*self.npi) * (b + c) / (b - c) / np.square(self.npi)


    def C_mn(self):
        a = self.tau * self.zeta * np.pi
        b = (a + self.Bi)*np.exp(a)
        c = (a - self.Bi)*np.exp(-1 * a)
        return 2 * np.sin(self.epsilon*self.mpi) * np.sin(self.gamma*self.npi) * (b + c) / (b - c) / (self.m * self.n * self.zeta * np.pi**3)
    
    
    def calculate_psi(self):
        self.zeta = np.sqrt(np.square(self.m) + np.square(self.n/self.beta))
        c_m0 = self.C_m0()
        c_0n = self.C_0n()
        c_mn = self.C_mn()
        self.psi = np.sqrt(self.epsilon * self.gamma / self.beta) * (np.sum(c_m0, 0)[0,:] + np.sum(c_0n, 1)[0,:] + np.sum(c_mn, (0, 1)))
        self.R_s = self.psi / self.k / np.sqrt(self.A_s)
        self.R_th = self.R_s + self.R_m + self.R_f
        return self.psi, self.R_s, self.R_th
    

class YovaSpreader(Spreader):

    def phi(self, eig:npt.NDArray):
        '''Spreading function for finite isotropic rectangular flux channel.'''
        n = (np.exp(2*eig*self._t) + 1) * eig - (1 - np.exp(2*eig*self._t))*self.Bi/self._t
        d = (np.exp(2*eig*self._t) - 1) * eig + (1 + np.exp(2*eig*self._t))*self.Bi/self._t
        return n / d


    def C_m(self):
        return np.square((np.sin(self._l1*self.sigma_m))) * self.phi(self.sigma_m) / self.sigma_m**3

    def C_n(self):
        return np.square((np.sin(self._w1*self.lamda_n))) * self.phi(self.lamda_n) / self.lamda_n**3

    def C_mn(self):
        n = np.square((np.sin(self._l1*self.sigma_m)))*np.square((np.sin(self._w1*self.lamda_n))) * self.phi(self.beta_mn)
        d = (self.sigma_m**2) * (self.lamda_n**2) * (self.beta_mn)
        return n / d


    def calculate_psi(self):
        self.sigma_m = self.mpi / self.l
        self.lamda_n = self.npi / self._w
        self.beta_mn = np.sqrt(np.square(self.sigma_m) + np.square(self.lamda_n))
        c_m = np.sum(self.C_m(), 0)[0, :] / 2 / (self.l1**2) / self.l / self.w / self.k
        c_n = np.sum(self.C_n(), 1)[0, :] / 2 / (self.w1**2) / self.l / self.w / self.k
        c_mn = np.sum(self.C_mn(), (0, 1)) / (self.l1**2) / (self.w1**2) / self.l / self.w / self.k
        self.R_s = c_m + c_n + c_mn
        self.psi = self.R_s * self.k * self.t
        self.R_th = self.R_s + self.R_m + self.R_f
        return self.psi, self.R_s, self.R_th


if __name__ == '__main__':
    L = 0.052
    W = 0.031
    L1 = 0.0195
    W1 = 0.0131
    t = np.linspace(0.001, 0.01, 101)
    # s = FengSpreader(
    #     L/2,
    #     W/2,
    #     L1/2,
    #     W1/2,
    #     t,
    #     15843
    # )
    # s.calculate_psi()
    # print(s)
    # print()
    s = YovaSpreader(
        L/2,
        W/2,
        L1/2,
        W1/2,
        t,
        15843
    )
    s.calculate_psi()
    t = np.squeeze(s.t)
    tau = np.squeeze(s.tau)
    Bi = np.squeeze(s.Bi)
    psi = np.squeeze(s.psi)
    A_s = np.squeeze(s.A_s)
    A_b = np.squeeze(s.A_b)

    R_tim = 0.0002 / 25 / A_s + 0.0002 / 25 / A_b

    R_1d = np.squeeze(s.R_m + s.R_f)
    R_s = np.squeeze(s.R_s)
    R_th = np.squeeze(s.R_th) + R_tim

    print(tabulate({
        'tau': tau,
        'Bi': Bi
    }, headers='keys'))

    plt.subplot(1, 2, 1)
    plt.plot(t, R_th, c='violet', label=r'$R_{th}$')
    plt.plot(t, R_tim, 'k--', label=r'$R_{tim}$')
    plt.plot(t, R_s, c='plum', label=r'$R_{s}$')
    plt.plot(t, R_1d, c='darkorange', label=r'$R_{1d}$')
    plt.grid(True, alpha=0.3)
    plt.xlabel(r'$t$ [m]')
    plt.ylabel(r'$R$ [K W-1]')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.loglog(tau, psi / t * np.sqrt(A_s), c='indigo', label=r'$\psi$')
    plt.grid(True, alpha=0.3)
    plt.xlabel(r'$\tau$')
    plt.legend()

    plt.show()