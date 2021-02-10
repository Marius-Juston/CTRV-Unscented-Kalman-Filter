import matplotlib.pyplot as plt
import numpy as np

from datapoint import DataPoint, DataType
from fusion_ukf import FusionUKF


def define_test_data():
    file = 'data/data.txt'

    ground_truth = []
    sensor_data = []

    with open(file, 'r') as file_data:
        for line in file_data.readlines():
            line = line.split()

            sensor_type = line.pop(0)

            if sensor_type == 'L':
                val1 = float(line.pop(0))
                val2 = float(line.pop(0))
                timestamp = float(line.pop(0)) / 1e6

                sensor_data.append(DataPoint(DataType.UWB, np.array([val1, val2]), timestamp))
            else:
                line.pop(0)
                line.pop(0)
                line.pop(0)
                timestamp = float(line.pop(0)) / 1e6

            x = float(line.pop(0))
            y = float(line.pop(0))
            vx = float(line.pop(0))
            vy = float(line.pop(0))
            yaw = float(line.pop(0))
            yaw_rate = float(line.pop(0))

            v = np.sqrt(vx ** 2 + vy ** 2)

            ground_truth.append(DataPoint(DataType.UWB, np.array([x, y, v, yaw, yaw_rate]), timestamp))

    return ground_truth, sensor_data


if __name__ == '__main__':
    ground_truth, sensor_data = define_test_data()

    filter = FusionUKF()

    ground_xy = np.array([[g.measurement_data[0], g.measurement_data[1]] for g in ground_truth])
    plt.plot(ground_xy[:, 0], ground_xy[:, 1], color=[1, 0, 0, 1])

    scatter_xy = np.array([[g.measurement_data[0], g.measurement_data[1]] for g in sensor_data])
    plt.scatter(scatter_xy[:, 0], scatter_xy[:, 1], color=[0, 0, 1, .5])

    x = []
    y = []

    for sensor in sensor_data:
        filter.update(sensor)

        x.append(filter.x[0])
        y.append(filter.x[1])

    plt.plot(x, y, color=[0, 1, 0, 1])

    plt.show()
