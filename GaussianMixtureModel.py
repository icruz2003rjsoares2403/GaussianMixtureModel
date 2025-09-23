
import time

import numpy as np

from scipy.special import psi
from scipy.special import betaln
from scipy.special import multigammaln

from scipy.special import softmax

from sklearn.cluster import kmeans_plusplus

class GaussianMixtureModel:

    def __init__(self, X : np.ndarray, M : int) -> None:
        
        self.X = X

        self.N, self.D = self.X.shape

        self.M = M

        self.alpha_0 = 1

        self.omega_0 = self.M

        self.tau_0 = 1

        self.mu_0 = np.zeros(shape = self.D)

        self.nu_0 = self.D + 2

        self.Phi_0 = np.identity(n = self.D)

        self.Psi_0 = np.linalg.inv(self.Phi_0)

        self.E_log_pi = np.zeros(shape = self.M)

        self.E_log_det_Lambda = np.zeros(shape = self.M)

        self.gamma = np.zeros(shape = (self.N, self.M))

        self.N_barra = np.zeros(shape = self.M)

        self.X_barra = np.zeros(shape = (self.M, self.D))

        self.S_barra = np.zeros(shape = (self.M, self.D, self.D))

        self.alpha = np.zeros(shape = self.M)

        self.omega = np.zeros(shape = self.M)

        self.tau = np.zeros(shape = self.M)

        self.mu = np.zeros(shape = (self.M, self.D))

        self.nu = np.zeros(shape = self.M)

        self.Phi = np.zeros(shape = (self.M, self.D, self.D))

        self.Psi = np.zeros(shape = (self.M, self.D, self.D))

        self.E_log_p_B = 0

        self.E_log_p_Z_mid_B = 0

        self.E_log_p_Lambda = 0

        self.E_log_p_mu_mid_Lambda = 0

        self.E_log_p_X_mid_Z_mu_Lambda = 0

        self.E_log_p = 0

        self.E_log_q_B = 0

        self.E_log_q_Z = 0

        self.E_log_q_Lambda = 0

        self.E_log_q_mu_mid_Lambda = 0

        self.E_log_q = 0

        self.ELBO = 0

        self.HISTORY = []

        self.Z = np.zeros(shape = self.N)

        self.pi = np.zeros(shape = self.M)

        self.Sigma = np.zeros(shape = (self.M, self.D, self.D))

        self.Lambda = np.zeros(shape = (self.M, self.D, self.D))

        self.ITER = 0

        self.Delta = 0

        self.RAE = 0

        self.CONVERGED = None

    def initialize_parameters(self) -> None:

        self.alpha = np.repeat(self.alpha_0, repeats = self.M)

        self.omega = np.repeat(self.omega_0, repeats = self.M)

        self.tau = np.repeat(self.tau_0, repeats = self.M)

        self.mu = kmeans_plusplus(self.X, n_clusters = self.M)[0]

        self.nu = np.repeat(self.nu_0, repeats = self.M)

        self.Phi = np.tile(self.Phi_0, reps = (self.M, 1, 1))

        self.Psi = np.tile(self.Psi_0, reps = (self.M, 1, 1))

    def update_E_log_pi(self) -> None:

        self.E_log_pi = psi(self.alpha) - psi(self.alpha + self.omega)

        self.E_log_pi += np.cumsum(psi(self.omega) - psi(self.alpha + self.omega))

        self.E_log_pi -= psi(self.omega) - psi(self.alpha + self.omega)

    def update_E_log_det_Lambda(self) -> None:

        self.E_log_det_Lambda = np.add.outer(self.nu, 1 - np.arange(self.D))/2

        self.E_log_det_Lambda = psi(self.E_log_det_Lambda).sum(axis = 1)

        self.E_log_det_Lambda += self.D*np.log(2) + np.linalg.slogdet(self.Psi)[1]

    def update_gamma(self) -> None:

        self.gamma = np.expand_dims(self.X, axis = 1) - np.expand_dims(self.mu, axis = 0)

        self.gamma = -self.nu*np.einsum('nmD, mDd, nmd -> nm', self.gamma, self.Psi, self.gamma)/2

        self.gamma += self.E_log_pi + self.E_log_det_Lambda/2 - self.D/(2*self.tau)

        self.gamma = softmax(self.gamma, axis = 1)

    def update_N_barra(self) -> None:

        self.N_barra = self.gamma.sum(axis = 0)

    def update_X_barra(self) -> None:

        self.X_barra = self.gamma.T @ self.X

        self.X_barra /= np.expand_dims(self.N_barra, axis = 1)

    def update_S_barra(self):

        self.S_barra = np.expand_dims(self.X, axis = 1) - np.expand_dims(self.X_barra, axis = 0)

        self.S_barra = np.einsum('nm, nmD, nmd -> mDd', self.gamma, self.S_barra, self.S_barra)

        self.S_barra /= np.expand_dims(self.N_barra, axis = (1, 2))

    def update_alpha(self) -> None:

        self.alpha = self.alpha_0 + self.N_barra

    def update_omega(self) -> None:

        self.omega = self.omega_0 + np.cumsum(self.N_barra[::-1])[::-1]

        self.omega -= self.N_barra

    def update_tau(self) -> None:

        self.tau = self.tau_0 + self.N_barra

    def update_mu(self) -> None:
    
        self.mu = np.expand_dims(self.N_barra, axis = 1)*self.X_barra

        self.mu += self.tau_0*self.mu_0

        self.mu /= np.expand_dims(self.tau, axis = 1)

    def update_nu(self) -> None:

        self.nu = self.nu_0 + self.N_barra

    def update_Phi(self) -> None:

        self.Phi = np.einsum('mD, md -> mDd', self.X_barra - self.mu_0, self.X_barra - self.mu_0)

        self.Phi *= np.expand_dims(self.tau_0*self.N_barra/self.tau, axis = (1, 2))

        self.Phi += np.expand_dims(self.N_barra, axis = (1, 2))*self.S_barra

        self.Phi += self.Phi_0

    def update_Psi(self) -> None:

        self.Psi = np.linalg.inv(self.Phi)

    def update_parameters(self) -> None:

        self.update_E_log_pi()

        self.update_E_log_det_Lambda()

        self.update_gamma()

        self.update_N_barra()

        self.update_X_barra()

        self.update_S_barra()

        self.update_alpha()

        self.update_omega()

        self.update_tau()

        self.update_mu()

        self.update_nu()

        self.update_Phi()

        self.update_Psi()

    def update_E_log_p_B(self) -> None:

        self.E_log_p_B = (self.alpha_0 - 1)*np.sum(psi(self.alpha) - psi(self.alpha + self.omega))

        self.E_log_p_B -= self.M*betaln(self.alpha_0, self.omega_0)

    def update_E_log_p_Z_mid_B(self) -> None:

        self.E_log_p_Z_mid_B = np.sum(self.gamma @ self.E_log_pi)

    def update_E_log_p_Lambda(self) -> None:

        self.E_log_p_Lambda = (self.nu_0 - self.D - 1)/2*self.E_log_det_Lambda

        self.E_log_p_Lambda -= self.nu*np.einsum('Dd, mDd -> m', self.Phi_0, self.Psi)/2

        self.E_log_p_Lambda -= self.nu_0*self.D/2*np.log(2) + self.nu_0/2*np.linalg.slogdet(self.Psi_0)[1]

        self.E_log_p_Lambda -= multigammaln(self.nu_0/2, self.D)

        self.E_log_p_Lambda = self.E_log_p_Lambda.sum()

    def update_E_log_p_mu_mid_Lambda(self) -> None:

        self.E_log_p_mu_mid_Lambda = self.D*np.log(self.tau_0) + self.E_log_det_Lambda

        self.E_log_p_mu_mid_Lambda -= self.D*np.log(2*np.pi) + self.D*self.tau_0/self.tau

        self.E_log_p_mu_mid_Lambda -= self.tau_0*self.nu*np.einsum('mD, mDd, md -> m', self.mu - self.mu_0, self.Phi, self.mu - self.mu_0)

        self.E_log_p_mu_mid_Lambda = self.E_log_p_mu_mid_Lambda.sum()/2

    def update_E_log_p_X_mid_Z_mu_Lambda(self) -> None:

        self.E_log_p_X_mid_Z_mu_Lambda = self.E_log_det_Lambda - self.D/self.tau - self.D*np.log(2*np.pi) 

        self.E_log_p_X_mid_Z_mu_Lambda -= self.nu*np.einsum('mD, mDd, md -> m', self.X_barra - self.mu, self.Psi, self.X_barra - self.mu)

        self.E_log_p_X_mid_Z_mu_Lambda -= self.nu*np.einsum('mDd, mDd -> m', self.S_barra, self.Psi)

        self.E_log_p_X_mid_Z_mu_Lambda = np.sum(self.N_barra*self.E_log_p_X_mid_Z_mu_Lambda)/2

    def update_E_log_p(self) -> None:

        self.update_E_log_p_B()

        self.E_log_p = self.E_log_p_B

        self.update_E_log_p_Z_mid_B()

        self.E_log_p += self.E_log_p_Z_mid_B

        self.update_E_log_p_Lambda()

        self.E_log_p += self.E_log_p_Lambda

        self.update_E_log_p_mu_mid_Lambda()

        self.E_log_p += self.E_log_p_mu_mid_Lambda

        self.update_E_log_p_X_mid_Z_mu_Lambda()

        self.E_log_p += self.E_log_p_X_mid_Z_mu_Lambda

    def update_E_log_q_B(self) -> None:

        self.E_log_q_B = (self.alpha + self.omega - 2)*psi(self.alpha + self.omega)

        self.E_log_q_B -= (self.alpha - 1)*psi(self.alpha) + (self.omega - 1)*psi(self.omega)

        self.E_log_q_B -= betaln(self.alpha, self.omega)

        self.E_log_q_B = self.E_log_q_B.sum()

    def update_E_log_q_Z(self) -> None:

        self.E_log_q_Z = np.log(self.gamma**self.gamma).sum()

    def update_E_log_q_Lambda(self) -> None:

        self.E_log_q_Lambda = (self.nu - self.D - 1)/2*self.E_log_det_Lambda - self.nu*self.D/2

        self.E_log_q_Lambda -= self.nu*self.D/2*np.log(2) + self.nu/2*np.linalg.slogdet(self.Psi)[1]

        self.E_log_q_Lambda -= multigammaln(self.nu/2, self.D)

        self.E_log_q_Lambda = self.E_log_q_Lambda.sum()

    def update_E_log_q_mu_mid_Lambda(self) -> None:

        self.E_log_q_mu_mid_Lambda = self.E_log_det_Lambda + self.D*np.log(self.tau)

        self.E_log_q_mu_mid_Lambda -= self.D*np.log(2*np.pi) + self.D

        self.E_log_q_mu_mid_Lambda = self.E_log_q_mu_mid_Lambda.sum()/2

    def update_E_log_q(self) -> None:

        self.update_E_log_q_B()

        self.E_log_q = self.E_log_q_B

        self.update_E_log_q_Z()

        self.E_log_q += self.E_log_q_Z

        self.update_E_log_q_Lambda()

        self.E_log_q += self.E_log_q_Lambda

        self.update_E_log_q_mu_mid_Lambda()

        self.E_log_q += self.E_log_q_mu_mid_Lambda

    def update_ELBO(self) -> None:

        self.update_E_log_p()

        self.ELBO = self.E_log_p

        self.update_E_log_q()

        self.ELBO -= self.E_log_q

        self.HISTORY.append(self.ELBO)

    def estimate_Z(self) -> None:

        self.Z = np.argmax(self.gamma, axis = 1)

    def estimate_pi(self) -> None:

        self.pi = self.alpha/(self.alpha + self.omega)

        self.pi *= np.cumprod(self.omega/(self.alpha + self.omega))

        self.pi /= self.omega/(self.alpha + self.omega)

    def estimate_Sigma(self) -> None:

        self.Sigma = self.Phi/(np.expand_dims(self.nu, axis = (1, 2)) - self.D - 1)

    def estimate_Lambda(self) -> None:

        self.Lambda = np.expand_dims(self.nu, axis = (1, 2))*self.Psi

    def estimate_parameters(self) -> None:

        self.estimate_Z()

        self.estimate_pi()

        self.estimate_Sigma()

        self.estimate_Lambda()

    def fit_parameters(self, MAX : int = 100, TOL : float = 1e-3) -> None:

        self.Delta = -time.time()

        self.initialize_parameters()

        for self.ITER in np.arange(start = 1, stop = MAX + 1):

            self.RAE = self.ELBO

            self.update_parameters()

            self.update_ELBO()

            self.RAE -= self.ELBO

            self.RAE = np.abs(self.RAE/self.ELBO)
            
            if self.RAE < TOL:

                break

        self.HISTORY = np.array(self.HISTORY)

        self.estimate_parameters()

        self.Delta += time.time()

        if self.RAE >= TOL:

            self.CONVERGED = False

            print('\nO algoritmo não convergiu\n')

        else:

            self.CONVERGED = True

            print(f'\nTempo Total em segundos: {self.Delta:0.5f}\n')

            print(f'Número Total de Iterações: {self.ITER}\n')

            print(f'Erro Absoluto Relativo: {self.RAE:0.5f}\n')
