"""
Diagnostic: load TB6 alone, draw ee_Link coordinate frame.
"""
import pybullet as p
import pybullet_data
import time, os
import numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_TB6 = os.path.join(SIM_DIR, "tb6.urdf")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0/240.0)
p.loadURDF("plane.urdf", [0, 0, 0])

tb6 = p.loadURDF(URDF_TB6, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
print(f"TB6 body id: {tb6}, joints: {p.getNumJoints(tb6)}")

# home pose (forward reaching)
home = [0.0, -0.70, 1.20, -2.30, -0.50, 0.0]
for i, a in enumerate(home):
    p.resetJointState(tb6, i, a)

for i in range(6):
    p.setJointMotorControl2(tb6, i, p.POSITION_CONTROL,
                            targetPosition=home[i], force=8000, maxVelocity=0.5)
for _ in range(300):
    p.stepSimulation()

# ee_Link is last link
flange_idx = p.getNumJoints(tb6) - 1
st = p.getLinkState(tb6, flange_idx)
pos, orn = st[0], st[1]

print(f"ee_Link (flange) index: {flange_idx}")
print(f"World position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
print(f"World quaternion: [{orn[0]:.4f}, {orn[1]:.4f}, {orn[2]:.4f}, {orn[3]:.4f}]")

# Draw ee_Link coordinate frame (0.1m axes)
R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
origin = np.array(pos)
for axis, color in [(0, [1, 0, 0]), (1, [0, 1, 0]), (2, [0, 0, 1])]:
    end = origin + R[:, axis] * 0.12
    p.addUserDebugLine(origin.tolist(), end.tolist(), color, 4)

# Labels: X=R, Y=G, Z=B
labels = ['+X', '+Y', '+Z']
for axis, color, label in [(0, [1, 0.3, 0.3], labels[0]),
                            (1, [0.3, 1, 0.3], labels[1]),
                            (2, [0.3, 0.3, 1], labels[2])]:
    pos_lbl = origin + R[:, axis] * 0.14
    p.addUserDebugText(label, pos_lbl.tolist(), color, 2.0)

print(f"\nFlange frame axes drawn (0.12m):")
print(f"  +X (red)   → flange local X")
print(f"  +Y (green) → flange local Y")
print(f"  +Z (blue)  → flange local Z (should point OUTWARD from arm)")

print("\nClose window to exit.")
try:
    while p.isConnected():
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
p.disconnect()
