from __future__ import annotations

__author__ = 'Drayrs'
__created__ = 240422

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class Spreader:

    def __init__(
            self,
            beta: float,
            epsilon: float,
            gamma: float,
            tau: float,
            Bi: float,
            l: float = 1,
            w: float = 1,
            k: float = 390,
            M: int = 40,
            N:int = 30
        ):  
        self.M = M
        self.N = N
        self.m, self.n, self.beta, self.epsilon, self.gamma, self.tau, self.Bi = np.meshgrid(
            np.arange(1, self.M + 1),
            np.arange(1, self.N + 1),
            beta,
            epsilon,
            gamma,
            tau,
            Bi,
            indexing='ij'
        )
        self.mpi = self.m * np.pi
        self.npi = self.n * np.pi
        self.zeta = np.sqrt(np.square(self.m) + np.square(self.n/self.beta))
        

    @classmethod
    def dimensioned(cls, l: float, w: float, t: float, l1: float, w1: float, h: float, k: float = 390) -> Spreader:
        '''Calculate non-dimensional volume lengths.
        
        Args:
            l: length of volume in x direction
            w: " y direction
            t: " z direction
            l1: length of projected heating area in x direction
            w1: " y direction
            h: heat transfer coefficient at bottom surface
        
        Returns:
            Non-dimensional lengths (beta, epsilon, gamma, tau)

        '''
        return cls(w/l, l1/l, w1/w, t/l, Bi=cls.Biot(h, t, k), l=l, w=w)
    

    @staticmethod
    def Biot(h: float, t: float, k: float):
        return h * t / k
    

    @property
    def h(self) -> float:
        '''Return the heat transfer coefficient at the bottom surface.
        
        '''
        return self.Bi * self.k / self.tau / self.l


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
    
    
    def R_th(self):
        c_m0 = self.C_m0()
        c_0n = self.C_0n()
        c_mn = self.C_mn()
        psi = np.sqrt(self.epsilon[0, 0, :] * self.gamma[0, 0, :] / self.beta[0, 0, :]) * (np.sum(c_m0, 0)[0,:] + np.sum(c_0n, 1)[0,:] + np.sum(c_mn, (0, 1)))
        return psi
    

    def __repr__(self) -> str:
        return f'{self.beta=}, {self.gamma=}, {self.epsilon=}, {self.tau=}, {self.Bi=}'


if __name__ == '__main__':
    spreader = Spreader(
        beta=[0.25, 0.5, 1],
        epsilon=0.25,
        gamma=[0.25, 0.75, 0.2],
        tau=np.logspace(-2, 1, 101),
        Bi=[0, 0.01, 0.1, 1, 10]
    )
    # spreader = Spreader.dimensioned(0.052, 0.031, 0.004, 0.0195, 0.0131, 15800)
    psi = spreader.R_th()
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
    param = [
        (0, 0),
        (0, 1),
        (1, 2),
        (1, 1),
        (2, 0),
        (2, 1)
    ]
    for i, ax in enumerate(axs.flatten()):
        for b in range(5):
            ax.loglog(
                spreader.tau[0, 0, param[i][0], 0, param[i][1], :, b],
                psi[param[i][0], 0, param[i][1], :, b],
                'k'
            )
    ax.set_xlim([0.01, 10])
    ax.set_ylim([0.001, 10])
    ax.set_xlabel(r'Dimensionless Spreader Thickness, $\tau$')
    ax.set_ylabel(r'Dimensionless Spreading Resis., $\psi$')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.show()