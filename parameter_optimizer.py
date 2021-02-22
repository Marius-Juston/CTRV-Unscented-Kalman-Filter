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
    # rmse = max_diff(estimations, ground_estimation)
    print(*rmse)

    return np.dot(rmse, weights)


def twiddle():
    ground_truth, sensor_data, first_timestamp = define_test_data()

    out = 'parameter_output.txt'
    sum_weighs = [1, 1, 1, 1, 1, 1]

    # Choose an initialization parameter vector
    p = [9.0001, 13.0, 10.0001, 15.9001, 3.0001, 1.0001, 2.0001, 4.9001, 5.9001, 1.0, 0, 0.0001, 0.0001, 0.0001, 2.0001,
         0.0001, 0.0001]
    p[10] = 0
    # Define potential changes
    dp = np.ones(17) * 1
    dp[9] = 1
    # Calculate the error
    best_err = run(sensor_data, ground_truth, first_timestamp, p, sum_weighs)

    threshold = 0.001

    print(best_err, ":", p)
    best_p = p

    with open(out, 'w') as out_file:
        out_file.write(f"{best_err}:" + ", ".join(map(str, best_p)) + "\n")
        out_file.flush()

        while np.all(dp > threshold):
            for i in range(len(p)):
                p[i] += dp[i]
                p[9] = np.clip(p[9], 0, 1)
                if i != 10 and p[i] < 0:
                    p[i] = 0.0001
                err = run(sensor_data, ground_truth, first_timestamp, p, sum_weighs)

                if err < best_err:  # There was some improvement
                    best_err = err
                    dp[i] *= 1.1

                    best_p = p

                    out_file.write(f"{best_err}:" + ", ".join(map(str, best_p)) + "\n")
                    out_file.flush()
                else:  # There was no improvement
                    p[i] -= 2 * dp[i]  # Go into the other direction
                    p[9] = np.clip(p[9], 0, 1)
                    if i != 10 and p[i] < 0:
                        p[i] = 0.0001
                    err = run(sensor_data, ground_truth, first_timestamp, p, sum_weighs)

                    if err < best_err:  # There was an improvement
                        best_err = err
                        dp[i] *= 1.05
                        best_p = p
                        out_file.write(f"{best_err}:" + ", ".join(map(str, best_p)) + "\n")
                        out_file.flush()
                    else:  # There was no improvement
                        p[i] += dp[i]
                        p[9] = np.clip(p[9], 0, 1)
                        if i != 10 and p[i] < 0:
                            p[i] = 0.0001
                        # As there was no improvement, the step size in either
                        # direction, the step size might simply be too big.
                        dp[i] *= 0.95

                print(best_err, ":", best_p, sum(dp), dp)

        # out_file.write(f"{best_err}:" + ", ".join(map(str, best_p)) + "\n")


if __name__ == '__main__':
    twiddle()
