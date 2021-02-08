import numpy as np

from util import normalize


class StateUpdater:
    def __init__(self, NX, N_SIGMA, WEIGHTS) -> None:
        super().__init__()
        self.N_SIGMA = N_SIGMA
        self.WEIGHTS = WEIGHTS
        self.NX = NX

    def compute_Tc(self, predicted_x, predicted_z, sigma_x, sigma_z):
        NZ = predicted_z.size

        Tc = np.zeros((self.NX, NZ))

        for i in range(self.N_SIGMA):
            dx = sigma_x[:, i] - predicted_x
            dx[3] = normalize(dx[3])
            dz = sigma_z[:, i] - predicted_z

            Tc += self.WEIGHTS[i] * np.matmul(np.atleast_2d(dx).transpose(), np.atleast_2d(dz))

        return Tc

    def update(self, z, S, Tc, predicted_z, predicted_x, predicted_P):
        Si = np.linalg.inv(S)
        K = np.matmul(Tc, Si)

        dz = z - predicted_z

        self.x = predicted_x + np.matmul(K, dz)
        self.P = predicted_P - np.matmul(K, np.matmul(S, K.transpose()))
        self.nis = np.matmul(dz.transpose(), np.matmul(Si, dz))

    def process(self, predicted_x, predicted_z, z, S, predicted_P, sigma_x, sigma_z):
        Tc = self.compute_Tc(predicted_x, predicted_z, sigma_x, sigma_z)
        self.update(z, S, Tc, predicted_z, predicted_x, predicted_P)
