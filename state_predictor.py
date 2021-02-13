import numpy as np

from util import normalize


class StatePredictor:
    def __init__(self, NX, N_SIGMA, N_AUGMENTED, VAR_SPEED_NOISE, VAR_YAW_RATE_NOISE, SCALE, WEIGHTS):
        super().__init__()

        self.WEIGHTS = WEIGHTS
        self.NX = NX
        self.SCALE = SCALE
        self.N_SIGMA = N_SIGMA
        self.N_AUGMENTED = N_AUGMENTED
        self.VAR_SPEED_NOISE = VAR_SPEED_NOISE
        self.VAR_YAW_RATE_NOISE = VAR_YAW_RATE_NOISE

        self.sigma = np.zeros((NX, N_SIGMA))
        self.x = np.zeros(NX)
        self.P = np.zeros((NX, N_SIGMA))

        self.YAW_RATE_THRESHOLD = 0.001

    def compute_augmented_sigma(self, x, P):
        augmented_sigma = np.zeros((self.N_AUGMENTED, self.N_SIGMA))
        augmented_x = np.zeros(self.N_AUGMENTED)
        augmented_P = np.zeros((self.N_AUGMENTED, self.N_AUGMENTED))

        augmented_x[:self.NX] = x
        augmented_P[:self.NX, :self.NX] = P
        augmented_P[self.NX, self.NX] = self.VAR_SPEED_NOISE
        augmented_P[self.NX + 1, self.NX + 1] = self.VAR_YAW_RATE_NOISE

        L = np.linalg.cholesky(augmented_P)
        augmented_sigma[:, 0] = augmented_x

        for c in range(self.N_AUGMENTED):
            i = c + 1
            augmented_sigma[:, i] = augmented_x + self.SCALE * L[:, c]
            augmented_sigma[:, i + self.N_AUGMENTED] = augmented_x - self.SCALE * L[:, c]

        return augmented_sigma

    def predict_sigma(self, augmented_sigma, dt):
        predicted_sigma = np.zeros((self.NX, self.N_SIGMA))

        for i in range(self.N_SIGMA):
            px = augmented_sigma[0, i]
            py = augmented_sigma[1, i]
            speed = augmented_sigma[2, i]
            yaw = augmented_sigma[3, i]
            yaw_rate = augmented_sigma[4, i]
            speed_noise = augmented_sigma[5, i]
            yaw_rate_noise = augmented_sigma[6, i]

            # PREDICT NEXT STEP USING CTRV Model

            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            dt_2 = dt * dt

            # Acceleration noise
            p_noise = 0.5 * speed_noise * dt_2
            y_noise = 0.5 * yaw_rate_noise * dt_2

            # Velocity change
            d_yaw = yaw_rate * dt
            d_speed = speed * dt

            # Predicted speed = constant speed + acceleration noise
            p_speed = speed + speed_noise * dt

            # Predicted yaw
            p_yaw = yaw + d_yaw + y_noise

            # Predicted yaw rate
            p_yaw_rate = yaw_rate + yaw_rate_noise * dt

            if abs(yaw_rate) <= self.YAW_RATE_THRESHOLD:
                p_px = px + d_speed * cos_yaw + p_noise * cos_yaw
                p_py = py + d_speed * sin_yaw + p_noise * sin_yaw
            else:
                k = speed / yaw_rate
                theta = yaw + d_yaw
                p_px = px + k * (np.sin(theta) - sin_yaw) + p_noise * cos_yaw
                p_py = py + k * (cos_yaw - np.cos(theta)) + p_noise * sin_yaw

            predicted_sigma[0, i] = p_px
            predicted_sigma[1, i] = p_py
            predicted_sigma[2, i] = p_speed
            predicted_sigma[3, i] = p_yaw
            predicted_sigma[4, i] = p_yaw_rate

            # ------------------

        return predicted_sigma

    def predict_x(self, predicted_sigma):
        predicted_x = np.zeros(self.NX)

        for i in range(self.N_SIGMA):
            predicted_x += self.WEIGHTS[i] * predicted_sigma[:, i]

        return predicted_x

    def predict_P(self, predicted_sigma, predicted_x):
        predicted_P = np.zeros((self.NX, self.NX))

        for i in range(self.N_SIGMA):
            dx = (predicted_sigma[:, i] - predicted_x)
            dx[3] = normalize(dx[3])
            predicted_P += self.WEIGHTS[i] * np.outer(dx, dx)

        return predicted_P

    def process(self, x, P, dt):
        augmented_sigma = self.compute_augmented_sigma(x, P)
        self.sigma = self.predict_sigma(augmented_sigma, dt)
        self.x = self.predict_x(self.sigma)
        self.P = self.predict_P(self.sigma, self.x)
