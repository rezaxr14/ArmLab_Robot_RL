# robot.py
import numpy as np

class Robot:
    """
    Represents a 3-link robotic arm.
    The arm has a waist, a shoulder, and an elbow joint.
    """
    def __init__(self, link_lengths):
        """
        Initializes the robot with given link lengths.
        
        Args:
            link_lengths (list or tuple): A list of three numbers representing
                                          the lengths of link 1, link 2, and link 3.
        """
        if len(link_lengths) != 3:
            raise ValueError("Please provide a list of 3 link lengths.")
        
        self.l1, self.l2, self.l3 = link_lengths
        self.thetas = [0, 0, 0] # Initial angles [waist, shoulder, elbow]

    def forward_kinematics(self, thetas):
        """
        Calculates the 3D position of each joint and the end-effector.
        
        Args:
            thetas (list or tuple): A list of three angles in radians for
                                   [waist (theta1), shoulder (theta2), elbow (theta3)].
        
        Returns:
            list of np.array: A list containing the 3D coordinates of
                              [joint0, joint1, joint2, end_effector].
        """
        self.thetas = thetas
        theta1, theta2, theta3 = self.thetas

        p0 = np.array([0, 0, 0])
        p1 = np.array([0, 0, self.l1])
        
        # Position of Joint 2 (elbow)
        x2 = self.l2 * np.cos(theta1) * np.sin(theta2)
        y2 = self.l2 * np.sin(theta1) * np.sin(theta2)
        z2 = self.l1 + self.l2 * np.cos(theta2)
        p2 = np.array([x2, y2, z2])
        
        # Position of End-Effector
        x3 = np.cos(theta1) * (self.l2 * np.sin(theta2) + self.l3 * np.sin(theta2 + theta3))
        y3 = np.sin(theta1) * (self.l2 * np.sin(theta2) + self.l3 * np.sin(theta2 + theta3))
        z3 = self.l1 + self.l2 * np.cos(theta2) + self.l3 * np.cos(theta2 + theta3)
        p3 = np.array([x3, y3, z3])

        return [p0, p1, p2, p3]

    def inverse_kinematics(self, target_pos, current_thetas):
        """
        Calculates the joint angles required to reach a target position,
        choosing the solution closest to the current configuration and handling singularities.
        
        Args:
            target_pos (np.array): The target [x, y, z] coordinates.
            current_thetas (np.array): The robot's current joint angles.
            
        Returns:
            list or None: A list of three angles [theta1, theta2, theta3] in radians,
                          or None if the target is unreachable.
        """
        x, y, z = target_pos
        
        # --- Step 1: Handle the waist angle (theta1) and singularity ---
        r_xy = np.sqrt(x**2 + y**2) # Horizontal distance to target
        
        # If the target is very close to the Z-axis (the singularity)
        if r_xy < 1e-6:
            # Don't calculate a new waist angle. Keep the current one.
            theta1 = current_thetas[0]
        else:
            # Calculate the waist angle as usual.
            theta1 = np.arctan2(y, x)
        
        # --- Step 2: Reduce to a 2D problem in the vertical plane ---
        r = r_xy # Use the same horizontal distance
        z_prime = z - self.l1
        
        D = np.sqrt(r**2 + z_prime**2)
        
        if D > self.l2 + self.l3 or D < abs(self.l2 - self.l3):
            return None
            
        # --- Step 3: Solve for both possible elbow and shoulder angles ---
        cos_theta3 = (D**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        
        # gamma is the target direction in the vertical plane, measured from
        # the +z axis exactly like theta2 in forward_kinematics().
        gamma = np.arctan2(r, z_prime)
        cos_alpha = (D**2 + self.l2**2 - self.l3**2) / (2 * D * self.l2)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)

        # Solution 1: Elbow Up
        theta3_up = np.arccos(cos_theta3)
        theta2_up = gamma - alpha
        sol1 = np.array([theta1, theta2_up, theta3_up])

        # Solution 2: Elbow Down
        theta3_down = -np.arccos(cos_theta3)
        theta2_down = gamma + alpha
        sol2 = np.array([theta1, theta2_down, theta3_down])

        # --- Step 4: Choose the solution with the smallest change ---
        diff1 = sol1 - current_thetas
        diff1 = (diff1 + np.pi) % (2 * np.pi) - np.pi
        cost1 = np.sum(diff1**2)

        diff2 = sol2 - current_thetas
        diff2 = (diff2 + np.pi) % (2 * np.pi) - np.pi
        cost2 = np.sum(diff2**2)

        if cost1 < cost2:
            return list(sol1)
        else:
            return list(sol2)

    @property
    def total_reach(self):
        """Calculates the maximum possible reach of the arm."""
        return self.l2 + self.l3
