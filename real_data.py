import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ukf.datapoint import DataType, DataPoint
from ukf.fusion_ukf import FusionUKF
from ukf.state import UKFState


def interpolation(v_initial, v_final, start_t, end_t, t):
    initial_theta = v_initial[UKFState.YAW]
    final_theta = v_final[UKFState.YAW]

    delta = ((final_theta - initial_theta) + np.pi) % (2 * np.pi) - np.pi
    out_theta = initial_theta + delta / (end_t - start_t) * (t - start_t)
    out_theta = (out_theta + np.pi) % (2 * np.pi) - np.pi

    out = v_initial + (v_final - v_initial) / (end_t - start_t) * (t - start_t)

    out[UKFState.YAW] = out_theta

    return out


def calculate_RMSE(estimations, ground_truths):
    rmse = np.zeros(estimations[0].size)

    for estimation, ground_truth in zip(estimations, ground_truths):
        diff = estimation - ground_truth
        diff = np.power(diff, 2)
        rmse += diff

    rmse /= len(estimations)
    rmse = np.sqrt(rmse)
    return rmse

def max_diff(estimations, ground_truths):
    max_error = np.zeros(estimations[0].size)

    for estimation, ground_truth in zip(estimations, ground_truths):
        diff = estimation - ground_truth
        diff[UKFState.YAW] %= (2 * np.pi)
        diff = abs(diff)

        max_error = np.fmax(max_error, diff)

    return max_error


def define_test_data():
    file = 'data/out.csv'

    ground_truth = []
    sensor_data = []

    first_timestamp = None

    with open(file, 'r') as file_data:
        for line in file_data.readlines():
            data = line.split(",")
            timestamp = int(data[-1])
            id = int(data[0])
            data = list(map(float, data[1:-1]))

            if first_timestamp is None:
                first_timestamp = timestamp

            if id == DataType.UWB:
                d, anchor_x, anchor_y, anchor_z, tag__offset_x, tag__offset_y, tag__offset_z = data

                sensor_data.append(
                    DataPoint(DataType.UWB, np.array(d), timestamp,
                              extra={
                                  'anchor': np.array([anchor_x, anchor_y, anchor_z]),
                                  'sensor_offset': np.array([tag__offset_x, tag__offset_y, tag__offset_z])
                              })
                )

            elif id == DataType.ODOMETRY:
                x, y, z, v, yaw, yaw_rate = data

                sensor_data.append(
                    DataPoint(DataType.ODOMETRY, np.array([x, y, z, v, yaw, yaw_rate]), timestamp)
                )

            elif id == DataType.GROUND_TRUTH:
                x, y, z, v, yaw, yaw_rate = data

                ground_truth.append(
                    DataPoint(DataType.GROUND_TRUTH, np.array([x, y, z, v, yaw, yaw_rate]), timestamp)
                )

            else:
                print("Other data")

    return ground_truth, sensor_data, first_timestamp


if __name__ == '__main__':
    ground_truth, sensor_data, first_timestamp = define_test_data()

    sensor_std = {
        DataType.LIDAR: {
            'std': [1, 1],
            'nz': 2
        },
        DataType.UWB: {
            'std': [.05],
            'nz': 1
        },
        DataType.ODOMETRY: {
            'std': [1, 1, .1, .1, .1, .1],
            'nz': 6
        }
    }

    print(len(ground_truth), len(sensor_data))

    fusion = FusionUKF(sensor_std,  4.9001, 6.9001, 1, 0)
    #
    # noise = np.array([
    #     np.random.normal(0, .15),
    #     np.random.normal(0, .15),
    #     np.random.normal(0, .05),
    #     np.random.normal(0, .1),
    #     np.random.normal(0, 0.0872665),
    #     np.random.normal(0, 0.05),
    #
    # ])
    noise = np.zeros(6)

    print(noise)

    fusion.initialize(ground_truth[0].measurement_data + noise, np.diag(
        [.0001, .0001, .0001, 2.0001, .0001,
         .0001]), first_timestamp)

    x = []
    y = []

    estimations = []

    current_g = ground_truth[0]
    next_g = ground_truth[1]

    i = 1

    ground_estimation = []

    for s in sensor_data:
        x.append(fusion.x[0])
        y.append(fusion.x[1])

        while s.timestamp > next_g.timestamp:
            current_g = ground_truth[i]
            next_g = ground_truth[i + 1]
            i += 1

        estimations.append(fusion.x)

        ground_estimation.append(
            interpolation(current_g.measurement_data, next_g.measurement_data, current_g.timestamp, next_g.timestamp,
                          s.timestamp))

        fusion.update(s)

    x.append(fusion.x[0])
    y.append(fusion.x[1])

    print(len(estimations), len(ground_estimation))
    print("X                   Y                  Z                   V                   YAW                YAW RATE")
    print(*calculate_RMSE(estimations, ground_estimation))

    ax: Axes = plt.gca()
    ax.set_aspect("equal")
    fig: Figure = ax.figure
    fig.tight_layout()

    ground_xy = np.array([[g.measurement_data[0], g.measurement_data[1]] for g in ground_truth])
    plt.plot(ground_xy[:, 0], ground_xy[:, 1], color=[1, 0, 0, 1], label='ground_truth')

    ground_xy = np.array([[g.measurement_data[0], g.measurement_data[1]] for g in sensor_data if g.data_type == DataType.ODOMETRY])
    plt.plot(ground_xy[:, 0], ground_xy[:, 1], color=[0, 0, 0, 1], label='odometry')

    plt.plot(x, y, color=[0, 1, 0, 1], label='fusion')

    ax.legend()
    plt.show()
