import numpy as np

from datapoint import DataPoint
from measurement_predictor import MeasurementPredictor
from state_predictor import StatePredictor


class FusionUKF:
    def __init__(self) -> None:
        super().__init__()

        self.initialized = False

        # Number of total states X, Y, velocity, yaw, yaw rate
        self.NX = 5

        # Settings values -----------------------------------
        self.N_AUGMENTED = self.NX + 2
        self.N_SIGMA = self.N_AUGMENTED * 2 + 1
        self.LAMBDA = 3 - self.N_AUGMENTED
        self.SCALE = np.sqrt(self.LAMBDA + self.N_AUGMENTED)
        self.W = 0.5 / (self.LAMBDA + self.N_AUGMENTED)
        self.W0 = self.LAMBDA / (self.LAMBDA + self.N_AUGMENTED)

        self.WEIGHTS = np.full(self.N_SIGMA, self.W)
        self.WEIGHTS[0] = self.W0
        # -----------------------------------

        # Uncertainty Settings -----------------------------------
        self.SPEED_NOISE_STD = 0.9
        self.YAW_RATE_NOISE_STD = 0.9

        self.SPEED_NOISE_VAR = self.SPEED_NOISE_STD ** 2
        self.YAW_RATE_NOISE_VAR = self.YAW_RATE_NOISE_STD ** 2
        # -----------------------------------

        # Measurement Uncertainty Settings -----------------------------------
        self.UWB_RANGE_NOISE = 0.257  # Meters
        self.UWB_RANGE_VAR = self.UWB_RANGE_NOISE ** 2
        # -----------------------------------

        self.x = np.zeros(self.NX)
        self.P = np.eyes(self.NX)

        self.state_predictor = StatePredictor(self.NX, self.N_SIGMA, self.N_AUGMENTED, self.SPEED_NOISE_VAR,
                                              self.YAW_RATE_NOISE_VAR, self.SCALE, self.WEIGHTS)

        self.measurement_predictor = MeasurementPredictor(self.UWB_RANGE_VAR)

    def initialize(self, x, initial_p, timestamp):
        self.x = x
        self.P = initial_p
        self.initialized = True
        self.timestamp = timestamp

    def update(self, data: DataPoint):
        predicted_z = None
        sigma_x = None
        sigma_z = None
        S = None

        dt = data.timestamp - self.timestamp  # seconds

        self.state_predictor.process(self.x, self.P, dt)
        self.x = self.state_predictor.x
        self.P = self.state_predictor.P
        sigma_x = self.state_predictor.sigma

        self.measurement_predictor.process(sigma_x, data.data_type)
