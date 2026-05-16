"""
Diagnostic: load Robotiq 3F alone at origin to verify URDF/meshes are OK.
"""
import pybullet as p
import pybullet_data
import time, os
import numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_ROBOTIQ = os.path.join(SIM_DIR, "robotiq_3f.urdf")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0/240.0)
p.loadURDF("plane.urdf", [0, 0, 0])

# load Robotiq at origin with identity orientation
rq = p.loadURDF(URDF_ROBOTIQ, [0, 0, 0.5], [0, 0, 0, 1], useFixedBase=True)

print(f"Robotiq body id: {rq}")
print(f"Number of joints: {p.getNumJoints(rq)}")
print(f"Base link mass: {p.getDynamicsInfo(rq, -1)[0]:.3f} kg")

for i in range(p.getNumJoints(rq)):
    info = p.getJointInfo(rq, i)
    name = info[1].decode()
    joint_type = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "FIXED", "PLANAR"][info[2]]
    lo, hi = info[8:10]
    print(f"  joint[{i}] {name:30s} type={joint_type:10s} "
          f"limits=[{lo:.3f}, {hi:.3f}]")

# Draw coordinate frame at palm origin
p.addUserDebugLine([0, 0, 0.5], [0.1, 0, 0.5], [1, 0, 0], 3)  # X red
p.addUserDebugLine([0, 0, 0.5], [0, 0.1, 0.5], [0, 1, 0], 3)  # Y green
p.addUserDebugLine([0, 0, 0.5], [0, 0, 0.6], [0, 0, 1], 3)    # Z blue

# Also draw at finger joint positions
for i in range(p.getNumJoints(rq)):
    info = p.getJointInfo(rq, i)
    name = info[1].decode()
    # Get joint origin in world frame (for fixed base, same as link frame)
    st = p.getLinkState(rq, i)
    pos = st[0]
    p.addUserDebugText(name[-12:], pos, [1, 1, 0], 1.5)

print("\nClose window to exit.  Check if Robotiq meshes render correctly.")
try:
    while p.isConnected():
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
p.disconnect()
