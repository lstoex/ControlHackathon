import casadi as ca
import control
from dataclasses import dataclass, field
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(local_path, ".."))

import models.droneXZModel as droneXZModel


@dataclass
class TwoDronesLQRCtrlConfig:
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.1, 0.1, 1.0, 0.1]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1]))
    Q_dist: float = 100.0  # Weight for distance constraint
    fixed_distance: float = 1.0  # Desired distance between drones
    n_drones: int = 2


class DroneXZTwoDronesLQRCtrl:
    def __init__(self, sampling_time: float, model: droneXZModel.DroneXZModel):
        self._sampling_time = sampling_time
        self._model = model
        self._ctrl_config = TwoDronesLQRCtrlConfig()
        self._goal_val = None

        # Create equilibrium states for both drones
        nx = self._model.model_config.nx
        self.x_equilibrium = np.zeros(nx * self._ctrl_config.n_drones)
        self.x_equilibrium[0] = 1.0  # Drone 1 x position
        self.x_equilibrium[1] = 1.0  # Drone 1 z position
        self.x_equilibrium[nx] = 2.0  # Drone 2 x position
        self.x_equilibrium[nx+1] = 1.0  # Drone 2 z position

        # Create equilibrium controls
        self.u_equilibrium = 0.5 * self._model.model_config.mass * self._model.model_config.gravity * np.ones((2 * self._ctrl_config.n_drones,))

        # Compute feedback gain
        self.fdbk_gain = self._compute_feedback_gain()

    def _compute_feedback_gain(self):
        nx = self._model.model_config.nx
        nu = self._model.model_config.nu
        
        # Linearize the system around equilibrium
        A1, B1 = self._model.linearizeDiscreteDynamics(
            self.x_equilibrium[:nx], 
            self.u_equilibrium[:nu]
        )
        A2, B2 = self._model.linearizeDiscreteDynamics(
            self.x_equilibrium[nx:], 
            self.u_equilibrium[nu:]
        )

        # Create block diagonal matrices for the combined system
        A = np.block([
            [A1, np.zeros((nx, nx))],
            [np.zeros((nx, nx)), A2]
        ])
        B = np.block([
            [B1, np.zeros((nx, nu))],
            [np.zeros((nx, nu)), B2]
        ])

        # Create augmented Q matrix to include distance constraint
        Q = np.kron(np.eye(self._ctrl_config.n_drones), self._ctrl_config.Q)
        
        # Add distance constraint to Q matrix
        Q[0, nx] = -self._ctrl_config.Q_dist  # x1-x2 coupling
        Q[1, nx+1] = -self._ctrl_config.Q_dist  # z1-z2 coupling
        Q[nx, 0] = -self._ctrl_config.Q_dist  # x2-x1 coupling
        Q[nx+1, 1] = -self._ctrl_config.Q_dist  # z2-z1 coupling
        
        Q[0, 0] += self._ctrl_config.Q_dist  # x1 diagonal
        Q[1, 1] += self._ctrl_config.Q_dist  # z1 diagonal
        Q[nx, nx] += self._ctrl_config.Q_dist  # x2 diagonal
        Q[nx+1, nx+1] += self._ctrl_config.Q_dist  # z2 diagonal

        # Create block diagonal R matrix
        R = np.kron(np.eye(self._ctrl_config.n_drones), self._ctrl_config.R)

        # Compute LQR gain
        K, S, E = control.dlqr(A, B, Q, R)
        return K

    def compute_LQR_control(self, x: np.ndarray):
        x_error = x - self.x_equilibrium
        u = self.u_equilibrium - self.fdbk_gain @ x_error
        return u

    @property
    def goal(self):
        return self._goal_val

    @goal.setter
    def goal(self, value):
        self._goal_val = value.flatten()[:, np.newaxis]
        # Update equilibrium positions based on new goal
        nx = self._model.model_config.nx
        self.x_equilibrium[0] = value[0]  # Drone 1 x
        self.x_equilibrium[1] = value[1]  # Drone 1 z
        self.x_equilibrium[nx] = value[2]  # Drone 2 x
        self.x_equilibrium[nx+1] = value[3]  # Drone 2 z
        # Recompute feedback gain for new equilibrium
        self.fdbk_gain = self._compute_feedback_gain()


def main():
    sampling_time = 0.05
    sim_length = 100
    
    # Initial states for both drones
    x1_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x2_init = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_init = np.concatenate([x1_init, x2_init])
    
    # Goal positions for both drones
    goal1 = np.array([1.0, 1.0])
    goal2 = np.array([2.0, 1.0])
    goal = np.concatenate([goal1, goal2])
    
    model = droneXZModel.DroneXZModel(sampling_time)
    controller = DroneXZTwoDronesLQRCtrl(sampling_time, model)
    controller.goal = goal
    
    x_trajectory, u_trajectory = model.simulateClosedLoop(sim_length, x_init, controller.compute_LQR_control, num_agents=2)
    
    # Custom animation for two drones
    def animate_two_drones(x_trajectory, u_trajectory, additional_lines_or_scatters=None):
        fontsize = 16
        params = {
            'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{cmbright}",
            'axes.labelsize': fontsize,
            'axes.titlesize': fontsize,
            'legend.fontsize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            "mathtext.fontset": "stixsans",
            "axes.unicode_minus": False,
        }
        matplotlib.rcParams.update(params)

        sim_length = u_trajectory.shape[1]
        fig, ax = plt.subplots()
        
        nx = model.model_config.nx
        nu = model.model_config.nu
        
        for i in range(sim_length+1):
            ax.set_aspect('equal')
            ax.set_xlim(-0.5, 3.0)
            ax.set_ylim(-0.5, 2.0)
            ax.set_xlabel('px(m)', fontsize=14)
            ax.set_ylabel('pz(m)', fontsize=14)
            
            # Plot trajectory for both drones
            ax.plot(x_trajectory[0, :i+1], x_trajectory[1, :i+1], color="tab:blue", linewidth=2, zorder=0, label="Drone 1 path")
            ax.plot(x_trajectory[nx, :i+1], x_trajectory[nx+1, :i+1], color="tab:red", linewidth=2, zorder=0, label="Drone 2 path")
            
            # Draw drones
            for drone_idx in range(2):
                offset = drone_idx * nx
                pos_x = x_trajectory[offset, i]
                pos_z = x_trajectory[offset+1, i]
                pitch = x_trajectory[offset+4, i]
                
                # Draw drone body
                left_x = pos_x - model.model_config.d * np.cos(pitch)
                left_z = pos_z - model.model_config.d * np.sin(pitch)
                right_x = pos_x + model.model_config.d * np.cos(pitch)
                right_z = pos_z + model.model_config.d * np.sin(pitch)
                
                color = "tab:blue" if drone_idx == 0 else "tab:red"
                ax.plot([left_x, right_x], [left_z, right_z], color=color, linewidth=5, zorder=1)
                ax.scatter(pos_x, pos_z, color=color, s=100, zorder=2)
                
                # Draw thrust vectors if not at the end
                if i < sim_length:
                    thrust_offset = drone_idx * nu
                    fl = u_trajectory[thrust_offset, i]
                    fr = u_trajectory[thrust_offset+1, i]
                    
                    patch_fl = patches.Arrow(left_x, left_z, 
                                           -0.1*fl*np.sin(pitch), 0.1*fl*np.cos(pitch), 
                                           color="tab:green", width=0.2)
                    patch_fr = patches.Arrow(right_x, right_z, 
                                           -0.1*fr*np.sin(pitch), 0.1*fr*np.cos(pitch), 
                                           color="tab:green", width=0.2)
                    ax.add_patch(patch_fl)
                    ax.add_patch(patch_fr)
            
            # Add goals if provided
            if additional_lines_or_scatters is not None:
                for key, value in additional_lines_or_scatters.items():
                    if value["type"] == "scatter":
                        ax.scatter(value["data"][0], value["data"][1], 
                                 color=value["color"], s=value["s"], 
                                 label=key, marker=value["marker"], zorder=3)
                    elif value["type"] == "line":
                        ax.plot(value["data"][0], value["data"][1], 
                              color=value["color"], linewidth=2, label=key)

            ax.set_title(f"Two Drones LQR Simulation: Step {i+1}")
            ax.legend()
            fig.subplots_adjust(bottom=0.15)
            plt.show(block=False)
            plt.pause(1.0 if i == sim_length else 0.2)
            ax.clear()
    
    # Prepare visualization data
    additional_lines_or_scatters = {
        "Goal 1": {"type": "scatter", "data": [[goal1[0]], [goal1[1]]], "color": "tab:orange", "s": 100, "marker": "x"},
        "Goal 2": {"type": "scatter", "data": [[goal2[0]], [goal2[1]]], "color": "tab:red", "s": 100, "marker": "x"}
    }
    
    animate_two_drones(x_trajectory, u_trajectory, additional_lines_or_scatters)
    return


if __name__ == "__main__":
    main() 