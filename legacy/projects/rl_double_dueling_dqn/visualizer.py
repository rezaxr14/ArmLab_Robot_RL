
# visualizer.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class RobotVisualizer:
    """
    A class to visualize a 3-link robotic arm in 3D.
    Modified to be controlled by the RL Environment.
    """
    def __init__(self, link_lengths):
        """
        Initializes the visualizer with robot's link lengths.
        
        Args:
            link_lengths (np.array): An array of three link lengths.
        """
        self.link_lengths = link_lengths
        
        plt.ion() # Turn on interactive mode for live plotting
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_xlabel('X-axis', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Y-axis', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Z-axis', fontsize=12, labelpad=10)
        self.ax.set_title('RL Robot Arm Control', fontsize=16, pad=20)
        
        max_reach = np.sum(self.link_lengths)
        self.ax.set_xlim([-max_reach, max_reach])
        self.ax.set_ylim([-max_reach, max_reach])
        self.ax.set_zlim([0, max_reach * 1.2])
        
        self.ax.view_init(elev=30., azim=45)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.links_plot, = self.ax.plot([], [], [], 'o-', lw=5, markersize=10, 
                                        markerfacecolor='deepskyblue', 
                                        markeredgecolor='navy', 
                                        color='royalblue')
        self.end_effector_plot, = self.ax.plot([], [], [], 'D', markersize=12, 
                                                color='gold', markeredgecolor='black')
        self.target_plot, = self.ax.plot([], [], [], '*', markersize=15, 
                                          color='red', markeredgecolor='darkred')
        self.fig.canvas.draw()
        plt.show(block=False)

    def plot_robot(self, joint_positions, target_pos=None):
        """
        Updates the plot with the new robot configuration.
        """
        xs = [p[0] for p in joint_positions]
        ys = [p[1] for p in joint_positions]
        zs = [p[2] for p in joint_positions]
        
        self.links_plot.set_data(xs, ys)
        self.links_plot.set_3d_properties(zs)
        
        end_effector = joint_positions[-1]
        self.end_effector_plot.set_data([end_effector[0]], [end_effector[1]])
        self.end_effector_plot.set_3d_properties([end_effector[2]])

        if target_pos is not None:
            self.target_plot.set_data([target_pos[0]], [target_pos[1]])
            self.target_plot.set_3d_properties([target_pos[2]])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Closes the plot window."""
        plt.close(self.fig)

