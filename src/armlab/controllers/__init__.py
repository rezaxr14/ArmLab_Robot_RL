"""Classical controllers for the ArmLab benchmark.

Available controllers:
- ``JointPositionController``: P control around an analytical IK solution.
- ``JointPDController``: PD control around an analytical IK solution.
- ``CartesianDLSController``: damped least-squares resolved-rate tracking
    with optional obstacle repulsion.
- ``ShootingMPCController``: random-shooting MPC on the kinematic model.
- ``DiscreteActionAdapter``: wraps any controller to emit discrete actions.
"""

from armlab.controllers.base import Controller, DiscreteActionAdapter, discrete_argmax
from armlab.controllers.cartesian import CartesianDLSController, SphereObstacle
from armlab.controllers.joint import JointPDController, JointPositionController
from armlab.controllers.mpc import ShootingMPCController
from armlab.controllers.planner import RRTPlannerController

__all__ = [
    "Controller",
    "DiscreteActionAdapter",
    "discrete_argmax",
    "CartesianDLSController",
    "SphereObstacle",
    "JointPDController",
    "JointPositionController",
    "RRTPlannerController",
    "ShootingMPCController",
]
