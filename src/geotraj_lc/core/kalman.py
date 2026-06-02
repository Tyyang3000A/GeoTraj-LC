import numpy as np


class LateralKalmanFilter:
    def __init__(self, init_d: float, dt: float, process_noise_q: float, measurement_noise_r: float):
        self.dt = dt

        self.X = np.array([[init_d], [0.0]], dtype=np.float64)
        self.F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64)

        self.Q = np.array(
            [[dt ** 4 / 4.0, dt ** 3 / 2.0], [dt ** 3 / 2.0, dt ** 2]],
            dtype=np.float64,
        ) * process_noise_q
        self.R = np.array([[measurement_noise_r]], dtype=np.float64)

    def predict(self) -> np.ndarray:
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.X

    def update(self, measurement_d: float):
        z = np.array([[measurement_d]], dtype=np.float64)
        y = z - self.H @ self.X
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)

        self.X = self.X + k @ y
        self.P = (np.eye(self.P.shape[0]) - k @ self.H) @ self.P

        return float(self.X[0, 0]), float(self.X[1, 0])

    def reset(self, new_d: float) -> None:
        self.X = np.array([[new_d], [0.0]], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64)

