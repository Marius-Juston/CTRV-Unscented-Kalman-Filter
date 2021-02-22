import time

import matplotlib.pyplot as plt
import numpy as np

from ukf.datapoint import DataPoint, DataType
from ukf.fusion_ukf import FusionUKF


def define_test_data():
    file = 'data/data.txt'

    ground_truth = []
    sensor_data = []
    sensor_offset_pose = []

    anchor_pos = np.array([5, 6, 0])
    anchor_pos2 = np.array([10, 20, 0])
    # anchor_pos2 = np.array([10,-15])
    # anchor_pos2 = np.array([1,2])

    sensor_offset = np.array([0, 2, 0])

    with open(file, 'r') as file_data:
        prev_yaw = None
        prev_d = None

        for line in file_data.readlines():
            line = line.split()

            sensor_type = line.pop(0)

            noise = np.random.normal(0, .25)

            if sensor_type == 'L':
                val1 = float(line.pop(0))
                val2 = float(line.pop(0))
                timestamp = float(line.pop(0)) / 1e6

                # sensor_data.append(DataPoint(DataType.LIDAR, np.array([val1, val2]), timestamp))

                x = float(line.pop(0))
                y = float(line.pop(0))
                vx = float(line.pop(0))
                vy = float(line.pop(0))
                yaw = float(line.pop(0))

                pose = np.array([x, y, 0]) + np.matmul(rotation_matrix(yaw), sensor_offset)

                distance = np.linalg.norm(pose - anchor_pos2) + noise
                # sensor_data.append(DataPoint(DataType.UWB, np.array([distance]), timestamp,
                #                              extra={"anchor": anchor_pos2, "sensor_offset": sensor_offset}))

                # print(yaw)

                v = np.sqrt(vx ** 2 + vy ** 2)

                if prev_d is None:
                    prev_d = timestamp
                if prev_yaw is None:
                    prev_yaw = yaw

                dt = timestamp - prev_d

                if dt > 0:
                    yaw_rate = (yaw - prev_yaw) / dt
                else:
                    yaw_rate = 0

                prev_yaw = yaw

                sensor_data.append(DataPoint(DataType.ODOMETRY, np.array([x, y, 0, v, yaw, yaw_rate]), timestamp,
                                             extra={"anchor": anchor_pos2, "sensor_offset": sensor_offset}))

            else:
                line.pop(0)
                line.pop(0)
                line.pop(0)

                timestamp = float(line.pop(0)) / 1e6

                if prev_d == 0:
                    prev_d = timestamp

                x = float(line.pop(0))
                y = float(line.pop(0))
                vx = float(line.pop(0))
                vy = float(line.pop(0))
                yaw = float(line.pop(0))

                pose = np.array([x, y, 0]) + np.matmul(rotation_matrix(yaw), sensor_offset)
                distance = np.linalg.norm(pose - anchor_pos) + noise

                sensor_data.append(DataPoint(DataType.UWB, np.array([distance]), timestamp,
                                             extra={"anchor": anchor_pos, "sensor_offset": sensor_offset}))

            yaw_rate = float(line.pop(0))

            v = np.sqrt(vx ** 2 + vy ** 2)

            ground_truth.append(DataPoint(DataType.LIDAR, np.array([x, y, v, yaw, yaw_rate]), timestamp))
            sensor_offset_pose.append(pose)

    return ground_truth, sensor_data, np.array(sensor_offset_pose)


def rotation_matrix(angle):
    s = np.sin(angle)
    c = np.cos(angle)

    return [[c, -s, 0],
            [s, c, 0],
            [0, 0, 1]]


if __name__ == '__main__':
    ground_truth, sensor_data, sensor_offset_pose = define_test_data()

    sensor_std = {
        DataType.LIDAR: {
            'std': [0.15, 0.15],
            'nz': 2
        },
        DataType.UWB: {
            'std': [.25],
            'nz': 1
        },
        DataType.ODOMETRY: {
            'std': [1, 1, 1, 1, 1, 1],
            'nz': 6
        }
    }

    filter = FusionUKF(sensor_std, .9, 1)

    ground_xy = np.array([[g.measurement_data[0], g.measurement_data[1]] for g in ground_truth])
    plt.plot(ground_xy[:, 0], ground_xy[:, 1], color=[1, 0, 0, 1])
    plt.plot(sensor_offset_pose[:, 0], sensor_offset_pose[:, 1], label="sensor", c='b')

    filter.initialize(np.array([*ground_truth[0].measurement_data[:2], 0, *ground_truth[0].measurement_data[2:]]),
                      np.eye(6) * 1,
                      ground_truth[0].timestamp)
    # scatter_xy = np.array([[g.measurement_data[0], g.measurement_data[1]] for g in sensor_data])
    # plt.scatter(scatter_xy[:, 0], scatter_xy[:, 1], color=[0, 0, 1, .5])

    x = []
    y = []

    time_ = []

    for sensor in sensor_data:
        s = time.time_ns()

        filter.update(sensor)

        time_.append(time.time_ns() - s)

        x.append(filter.x[0])
        y.append(filter.x[1])

    time_ = np.array(time_)

    print(np.mean(time_), np.std(time_), np.sum(time_), len(time_), np.mean(time_[time_ <= 1.4 * 1e7]))

    plt.plot(x, y, color=[0, 1, 0, 1])

    plt.show()

    plt.plot(time_)
    plt.show()
