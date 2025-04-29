import casadi as ca
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
class TwoDronesCtrlConfig:
    max_fl: float = 15.0
    max_fr: float = 15.0
    fixed_distance: float = 0.1  # Desired distance between drones
    n_drones: int = 2

    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.1, 0.1, 1.0, 0.1]))
    Q_e: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 1.0, 1.0, 10.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01]))
    Q_dist: float = 10.0  # Weight for distance constraint
    n_hrzn = 10


class DroneXZTwoDronesCtrl:
    def __init__(self, sampling_time: float, model: droneXZModel.DroneXZModel):
        self._sampling_time = sampling_time
        self._model = model
        self._ctrl_config = TwoDronesCtrlConfig()
        self.x_sol = None
        self.u_sol = None
        self._goal_val = None

    def _define_ocp_variables(self):
        # State variables for both drones
        self._x_opt = ca.SX.sym('x', self._ctrl_config.n_drones * self._model.model_config.nx, self._ctrl_config.n_hrzn+1)
        # Control variables for both drones
        self._u_opt = ca.SX.sym('u', self._ctrl_config.n_drones * self._model.model_config.nu, self._ctrl_config.n_hrzn)
        # Parameters: initial state and goal positions
        self._x_0_param = ca.SX.sym('x_0', self._ctrl_config.n_drones * self._model.model_config.nx)
        self._goal_param = ca.SX.sym('goal', 2 * self._ctrl_config.n_drones)  # [x1, z1, x2, z2]

    def _setup_constraints(self):
        self._g = []
        self._lbg = []
        self._ubg = []
        
        # Input constraints for both drones
        for i in range(self._ctrl_config.n_hrzn):
            # Drone 1
            self._g.append(self._u_opt[0, i])
            self._lbg.append(0.0)
            self._ubg.append(self._ctrl_config.max_fl)
            self._g.append(self._u_opt[1, i])
            self._lbg.append(0.0)
            self._ubg.append(self._ctrl_config.max_fr)
            # Drone 2
            self._g.append(self._u_opt[2, i])
            self._lbg.append(0.0)
            self._ubg.append(self._ctrl_config.max_fl)
            self._g.append(self._u_opt[3, i])
            self._lbg.append(0.0)
            self._ubg.append(self._ctrl_config.max_fr)

        # System dynamics for both drones
        self._g.append(self._x_opt[:, 0] - self._x_0_param)
        self._lbg.append(np.zeros(2 * self._model.model_config.nx))
        self._ubg.append(np.zeros(2 * self._model.model_config.nx))

        for i in range(self._ctrl_config.n_hrzn):
            # Drone 1 dynamics
            x1_next = self._model.f_disc(self._x_opt[:self._model.model_config.nx, i], 
                                       self._u_opt[:self._model.model_config.nu, i])
            self._g.append(self._x_opt[:self._model.model_config.nx, i+1] - x1_next)
            self._lbg.append(np.zeros(self._model.model_config.nx))
            self._ubg.append(np.zeros(self._model.model_config.nx))
            
            # Drone 2 dynamics
            x2_next = self._model.f_disc(self._x_opt[self._model.model_config.nx:, i], 
                                       self._u_opt[self._model.model_config.nu:, i])
            self._g.append(self._x_opt[self._model.model_config.nx:, i+1] - x2_next)
            self._lbg.append(np.zeros(self._model.model_config.nx))
            self._ubg.append(np.zeros(self._model.model_config.nx))

            # Distance constraint between drones (using squared distance)
            pos1 = self._x_opt[:2, i]
            pos2 = self._x_opt[self._model.model_config.nx:self._model.model_config.nx+2, i]
            dx = pos1[0] - pos2[0]
            dy = pos1[1] - pos2[1]
            squared_distance = dx**2 + dy**2
            self._g.append(squared_distance)
            self._lbg.append(self._ctrl_config.fixed_distance**2)
            self._ubg.append(np.inf)

        # Add terminal velocity constraints to prevent overshooting
        nx = self._model.model_config.nx
        # Drone 1 terminal velocity constraints
        self._g.append(self._x_opt[2, -1])  # vx1
        self._g.append(self._x_opt[3, -1])  # vz1
        self._lbg.extend([-0.1, -0.1])  # Small terminal velocities
        self._ubg.extend([0.1, 0.1])
        
        # Drone 2 terminal velocity constraints
        self._g.append(self._x_opt[nx+2, -1])  # vx2
        self._g.append(self._x_opt[nx+3, -1])  # vz2
        self._lbg.extend([-0.1, -0.1])  # Small terminal velocities
        self._ubg.extend([0.1, 0.1])

    def _setup_obj_func(self):
        self._J = 0.0
        nx = self._model.model_config.nx
        
        # Create goal states for both drones
        x1_goal = ca.vertcat(self._goal_param[:2], ca.DM.zeros(4, 1))
        x2_goal = ca.vertcat(self._goal_param[2:], ca.DM.zeros(4, 1))
        
        # Equilibrium control inputs
        u_equilibrium = 0.5 * self._model.model_config.mass * self._model.model_config.gravity * ca.DM.ones(2, 1)
        
        for i in range(self._ctrl_config.n_hrzn):
            # Cost for drone 1
            self._J += (self._x_opt[:nx, i] - x1_goal).T @ self._ctrl_config.Q @ (self._x_opt[:nx, i] - x1_goal)
            self._J += (self._u_opt[:2, i] - u_equilibrium).T @ self._ctrl_config.R @ (self._u_opt[:2, i] - u_equilibrium)
            
            # Cost for drone 2
            self._J += (self._x_opt[nx:, i] - x2_goal).T @ self._ctrl_config.Q @ (self._x_opt[nx:, i] - x2_goal)
            self._J += (self._u_opt[2:, i] - u_equilibrium).T @ self._ctrl_config.R @ (self._u_opt[2:, i] - u_equilibrium)
            
            # Distance maintenance cost (using squared distance)
            # pos1 = self._x_opt[:2, i]
            # pos2 = self._x_opt[nx:nx+2, i]
            # dx = pos1[0] - pos2[0]
            # dy = pos1[1] - pos2[1]
            # squared_distance = dx**2 + dy**2
            # self._J += self._ctrl_config.Q_dist * (squared_distance - self._ctrl_config.fixed_distance**2)**2

        # Terminal costs with higher weights
        self._J += (self._x_opt[:nx, -1] - x1_goal).T @ self._ctrl_config.Q_e @ (self._x_opt[:nx, -1] - x1_goal)
        self._J += (self._x_opt[nx:, -1] - x2_goal).T @ self._ctrl_config.Q_e @ (self._x_opt[nx:, -1] - x2_goal)

    def setup_OCP(self):
        self._define_ocp_variables()
        self._setup_constraints()
        self._setup_obj_func()
        ocp = {
            'x': ca.veccat(self._x_opt, self._u_opt),
            'p': ca.vertcat(self._x_0_param, self._goal_param),
            'g': ca.vertcat(*self._g),
            'f': self._J
        }
        opts = {'ipopt': {'print_level': 0, 'max_iter': 1000}, "print_time": False}
        self._solver = ca.nlpsol('solver', 'ipopt', ocp, opts)

    def solve_OCP(self, x_0: np.ndarray):
        x_0 = x_0.flatten()[:, np.newaxis]
        if self.x_sol is None:
            self.x_sol = np.tile(x_0, (1, self._ctrl_config.n_hrzn + 1))
        if self.u_sol is None:
            self.u_sol = np.zeros((2 * self._model.model_config.nu, self._ctrl_config.n_hrzn))
        if self._goal_val is None:
            raise ValueError("Goal value must be set before solving the OCP.")
            
        solution = self._solver(
            x0=ca.vertcat(self.x_sol.flatten(), self.u_sol.flatten()),
            p=ca.vertcat(x_0, self._goal_val),
            lbg=ca.vertcat(*self._lbg),
            ubg=ca.vertcat(*self._ubg)
        )
        
        nx = self._model.model_config.nx
        nu = self._model.model_config.nu
        n_hrzn = self._ctrl_config.n_hrzn
        
        x_sol = solution['x'].full().flatten()[:(n_hrzn+1) * 2 * nx].reshape((2 * nx, n_hrzn+1), order='F')
        u_sol = solution['x'].full().flatten()[(n_hrzn+1) * 2 * nx:].reshape((2 * nu, n_hrzn), order='F')
        
        self.x_sol = x_sol
        self.u_sol = u_sol

        return u_sol[:, 0]

    @property
    def goal(self):
        return self._goal_val

    @goal.setter
    def goal(self, value):
        self._goal_val = value.flatten()[:, np.newaxis]


def main():
    sampling_time = 0.05
    sim_length = 70
    
    # Initial states for both drones
    x1_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x2_init = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_init = np.concatenate([x1_init, x2_init])
    
    # Goal positions for both drones
    goal1 = np.array([1.0, 1.0])
    goal2 = np.array([2.0, 1.0])
    goal = np.concatenate([goal1, goal2])
    
    model = droneXZModel.DroneXZModel(sampling_time)
    controller = DroneXZTwoDronesCtrl(sampling_time, model)
    controller.setup_OCP()
    controller.goal = goal
    
    x_trajectory, u_trajectory = model.simulateClosedLoop(sim_length, x_init, controller.solve_OCP, num_agents=2)
    
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

            ax.set_title(f"Two Drones Simulation: Step {i+1}")
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