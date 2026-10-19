# main.py
import numpy as np
import time
from robot import Robot
from visualizer import RobotVisualizer

def main():
    link_lengths = [5, 10, 8]
    robot = Robot(link_lengths)
    visualizer = RobotVisualizer(robot)
    
    print("Starting inverse kinematics simulation...")
    print("The robot will try to follow a moving target smoothly.")
    print("Close the plot window to exit.")
    
    t = 0
    current_thetas = np.array(robot.thetas, dtype=float) 
    smoothing_factor = 0.6

    # Change the loop condition to use the flag from the visualizer
    while visualizer.is_running:
        radius = 12
        speed = 0.3 
        height = 12
        target_pos = np.array([
            radius * np.cos(t * speed), 
            radius * np.sin(t * speed), 
            height + 4 * np.sin(t * speed * 2)
        ])
        
        ik_thetas = robot.inverse_kinematics(target_pos, current_thetas)
        
        if ik_thetas is not None:
            target_thetas_np = np.array(ik_thetas)
            
            error = target_thetas_np - current_thetas
            error = (error + np.pi) % (2 * np.pi) - np.pi
            
            current_thetas = current_thetas + smoothing_factor * error
        
        joint_positions = robot.forward_kinematics(current_thetas)
        visualizer.plot_robot(joint_positions, target_pos)
        
        t += 0.02
    
    print("Simulation finished.")

if __name__ == '__main__':
    main()