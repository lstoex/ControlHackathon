import casadi as ca
import control
from dataclasses import dataclass
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(local_path, ".."))

import models.droneXZModel as droneXZModel

"""
x = (px, pz, pitch, vx, vz, vpitch)
a = (fl, fr)
"""

@dataclass
class DroneXZLQRCtrlConfig:
    Q: np.ndarray = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    R: np.ndarray = np.diag([0.1, 0.1])


class DroneXZLQRCtrl:
    def __init__(self, sampling_time: float, model: droneXZModel.DroneXZModel):
        self._sampling_time = sampling_time
        self._model = model
        self._ctrl_config = DroneXZLQRCtrlConfig()
        self._goal_val = None

        def compute_feedback_gain(x_equilibrium: np.ndarray, u_equilibrium: np.ndarray):
            A, B = self._model.linearizeContinuousDynamics(x_equilibrium, u_equilibrium)
            Q = self._ctrl_config.Q
            R = self._ctrl_config.R
            K, S, E = control.lqr(A, B, Q, R)
            return K

        self.x_equilibrium = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.u_equilibrium = np.array([self._model.model_config.mass*self._model.model_config.gravity, self._model.model_config.mass*self._model.model_config.gravity])
        self.fdbk_gain = compute_feedback_gain(self.x_equilibrium, self.u_equilibrium)


    def compute_LQR_control(self, x: np.ndarray,):
        x_error = x - self.x_equilibrium
        u = self.u_equilibrium - self.fdbk_gain @ x_error
        return u


def main():
    sampling_time = 0.05
    sim_length = 30
    x_init = np.array([0.02, 0.02, -np.pi/20., 0.1, -0.05, 0.0])
    goal = 2.0
    model = droneXZModel.DroneXZModel(sampling_time)
    controller = DroneXZLQRCtrl(sampling_time, model)
    controller.goal = goal
    x_trajectory, u_trajectory = model.simulateClosedLoop(sim_length, x_init, controller.compute_LQR_control)
    model.animateSimulation(x_trajectory, u_trajectory)
    return


if __name__ == "__main__":
    main()