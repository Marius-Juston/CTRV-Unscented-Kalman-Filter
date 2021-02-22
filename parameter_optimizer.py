import numpy as np

from real_data import define_test_data, interpolation, calculate_RMSE
from ukf.datapoint import DataType
from ukf.fusion_ukf import FusionUKF


def run(sensor_data, ground_truth, first_timestamp, parameters, weights=(1, 1, 1, 1, 1, 1)):
    sensor_std = {
        DataType.UWB: {
            'std': [parameters[0]],
            'nz': 1
        },
        DataType.ODOMETRY: {
            'std': [parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6]],
            'nz': 6
        }
    }

    fusion = FusionUKF(sensor_std, parameters[7], parameters[8], parameters[9], parameters[10])

    fusion.initialize(ground_truth[0].measurement_data, np.diag(
        [parameters[11], parameters[12], parameters[13], parameters[14], parameters[15], parameters[16]]),
                      first_timestamp)

    estimations = []

    current_g = ground_truth[0]
    next_g = ground_truth[1]

    i = 1

    ground_estimation = []

    for s in sensor_data:
        while s.timestamp > next_g.timestamp:
            current_g = ground_truth[i]
            next_g = ground_truth[i + 1]
            i += 1

        estimations.append(fusion.x)

        ground_estimation.append(
            interpolation(current_g.measurement_data, next_g.measurement_data, current_g.timestamp, next_g.timestamp,
                          s.timestamp))

        try:
            fusion.update(s)
        except np.linalg.LinAlgError as e:
            print(e)
            return float('inf')

    print("X                   Y                  Z                   V                   YAW                YAW RATE")
    rmse = calculate_RMSE(estimations, ground_estimation)
    print(*rmse)

    return np.dot(rmse, weights)


def twiddle():
    ground_truth, sensor_data, first_timestamp = define_test_data()

    # Choose an initialization parameter vector
    p = np.ones(17)
    p[10] = 0
    # Define potential changes
    dp = np.ones(17)
    # Calculate the error
    best_err = run(sensor_data, ground_truth, first_timestamp, p)

    threshold = 0.001

    print(best_err, ":", p)

    while sum(dp) > threshold:
        for i in range(len(p)):
            p[i] += dp[i]
            p[9] = np.clip(p[9], 0, 1)
            err = run(sensor_data, ground_truth, first_timestamp, p)

            if err < best_err:  # There was some improvement
                best_err = err
                dp[i] *= 1.1
            else:  # There was no improvement
                p[i] -= 2 * dp[i]  # Go into the other direction
                p[9] = np.clip(p[9], 0, 1)
                err = run(sensor_data, ground_truth, first_timestamp, p)

                if err < best_err:  # There was an improvement
                    best_err = err
                    dp[i] *= 1.05
                else:  # There was no improvement
                    p[i] += dp[i]
                    p[9] = np.clip(p[9], 0, 1)
                    # As there was no improvement, the step size in either
                    # direction, the step size might simply be too big.
                    dp[i] *= 0.95

            print(best_err, ":", p)


if __name__ == '__main__':
    twiddle()
