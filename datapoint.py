class DataType:
    UWB = 1
    ODOMETRY = 2


class DataPoint:
    def __init__(self, data_type, measurement_data, timestamp) -> None:
        super().__init__()
        self.data_type = data_type
        self.measurement_data = measurement_data
        self.timestamp = timestamp
