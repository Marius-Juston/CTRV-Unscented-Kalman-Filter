import numpy as np

from datapoint import DataType


class MeasurementPredictor:
    def __init__(self, UWB_RANGE_VAR, N_SIGMA, WEIGHTS) -> None:
        super().__init__()

        self.WEIGHTS = WEIGHTS
        self.N_SIGMA = N_SIGMA
        self.UWB_RANGE_VAR = UWB_RANGE_VAR

        self.z = None
        self.S = None
        self.sigma_z = None
        self.current_type = None

        self.R = None
        self.nz = None

        self.R_UWB = np.array([[UWB_RANGE_VAR, 0],
                               [0, UWB_RANGE_VAR]])

    def initialize(self, sensor_type):
        self.current_type = sensor_type

        if sensor_type == DataType.UWB:
            self.R = self.R_UWB
            self.nz = 2

    def compute_sigma_z(self, sigma_x):
        THRESHOLD = 1e-4

        sigma = np.zeros((self.nz, self.N_SIGMA))

        for i in range(self.N_SIGMA):
            if self.current_type == DataType.UWB:
                sigma[0, i] = sigma_x[0, i]  # py
                sigma[1, i] = sigma_x[1, i]  # px

        return sigma

    def compute_z(self, sigma):
        z = np.zeros(self.nz)

        for i in range(self.N_SIGMA):
            z += self.WEIGHTS[i] * sigma[:, i]

        return z

    def compute_S(self, sigma, z):
        S = np.zeros((self.nz, self.nz))

        for i in range(self.N_SIGMA):
            dz = sigma[:, i] - z

            S += self.WEIGHTS[i] * np.matmul(dz, dz.transpose())

        S += self.R

        return S

    def process(self, sigma_x, sensor_type):
        self.initialize(sensor_type)
        self.sigma_z = self.compute_sigma_z(sigma_x)
        self.z = self.compute_z(self.sigma_z)
        self.S = self.compute_S(self.sigma_z, self.z)
