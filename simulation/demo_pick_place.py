"""
Pick-and-place demo: TB6 R5 + Robotiq 3F gripper.

Architecture:
  - TB6 joints:  resetJointState at 60 Hz (smooth, no windup, no gravity sag)
  - Robotiq:     synced to flange EVERY physics step (never falls)
  - Cube:        teleported into gripper at grasp, constraint-follows thereafter
  - IK used for grasp pose so palm actually reaches the cube
"""

import pybullet as p, pybullet_data, time, os, numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_TB6     = os.path.join(SIM_DIR, "tb6.urdf")
URDF_ROBOTIQ = os.path.join(SIM_DIR, "robotiq_3f.urdf")

# ---- mounting (palm → TB6 flange) ----
MOUNT_TRANS = [0.0, 0.0, 0.048]
MOUNT_RPY   = [np.pi/2, 0.0, 0.0]

# ---- scene ----
PLATFORM_Z = -0.06
BLOCK_HALF = 0.022
CUBE_Z     = PLATFORM_Z + BLOCK_HALF          # -0.038
PICK_XY    = [0.50, 0.0]

HOME = np.array([0.0, -0.70, 1.20, -2.30, -0.50, 0.0])

# ---- finger joints ----
FB = ['palm_finger_1_joint', 'palm_finger_2_joint']
FK = ['finger_1_joint_1','finger_1_joint_2','finger_1_joint_3',
      'finger_2_joint_1','finger_2_joint_2','finger_2_joint_3',
      'finger_middle_joint_1','finger_middle_joint_2','finger_middle_joint_3']

# =====================================================================
def make_platform():
    h = [0.18, 0.15, 0.04]   # smaller, won't block arm links
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=h, rgbaColor=[0.5,0.3,0.2,1])
    c = p.createCollisionShape(p.GEOM_BOX, halfExtents=h)
    return p.createMultiBody(0,c,v,[PICK_XY[0],PICK_XY[1],PLATFORM_Z-h[2]],[0,0,0,1])

def make_cube():
    h = BLOCK_HALF
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=[h,h,h], rgbaColor=[0.95,0.2,0.15,1])
    c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[h,h,h])
    return p.createMultiBody(0.05,c,v,[PICK_XY[0],PICK_XY[1],CUBE_Z],[0,0,0,1])

# =====================================================================
def load_tb6():
    tb6 = p.loadURDF(URDF_TB6, [0,0,0], [0,0,0,1], useFixedBase=True)
    for i, a in enumerate(HOME):
        p.resetJointState(tb6, i, a)
    for _ in range(10): p.stepSimulation()
    return tb6

def load_robotiq():
    """Load Robotiq at origin.  Will be synced manually every step."""
    rq = p.loadURDF(URDF_ROBOTIQ, [0,0,0], [0,0,0,1], useFixedBase=False)
    jidx = {}
    for i in range(p.getNumJoints(rq)):
        name = p.getJointInfo(rq, i)[1].decode()
        jidx[name] = i
        p.changeDynamics(rq, i, mass=0.001, jointDamping=0.0)
        lo, _ = p.getJointInfo(rq, i)[8:10]
        p.resetJointState(rq, i, lo)
    p.changeDynamics(rq, -1, mass=0.001)
    set_gripper(rq, jidx, 'open')
    return rq, jidx

def sync_rq(tb6, fi, rq):
    """Place Robotiq at the exact flange pose (no physics constraint needed)."""
    fs = p.getLinkState(tb6, fi)
    mq = p.getQuaternionFromEuler(MOUNT_RPY)
    rpos, rorn = p.multiplyTransforms(fs[0], fs[1], MOUNT_TRANS, mq)
    p.resetBasePositionAndOrientation(rq, rpos, rorn)

def set_gripper(rq, jidx, action):
    f, mv = 3.0, 0.5
    if action == 'close':
        for k in FB:
            if k in jidx:
                _, hi = p.getJointInfo(rq, jidx[k])[8:10]
                p.setJointMotorControl2(rq, jidx[k], p.POSITION_CONTROL,
                                        targetPosition=hi, force=f, maxVelocity=mv)
        for k in FK:
            if k in jidx:
                lo, hi = p.getJointInfo(rq, jidx[k])[8:10]
                p.setJointMotorControl2(rq, jidx[k], p.POSITION_CONTROL,
                                        targetPosition=lo+0.8*(hi-lo), force=f, maxVelocity=mv)
    else:
        for k in FB+FK:
            if k in jidx:
                lo, _ = p.getJointInfo(rq, jidx[k])[8:10]
                p.setJointMotorControl2(rq, jidx[k], p.POSITION_CONTROL,
                                        targetPosition=lo, force=f, maxVelocity=mv)

# =====================================================================
def move_arm(tb6, fi, rq, targets, duration=1.5):
    """Move TB6 joints + sync Robotiq at 240 Hz.  No physics constraint."""
    targets = np.asarray(targets, dtype=float)
    current = np.array([p.getJointState(tb6, i)[0] for i in range(6)])
    steps = int(duration * 240)
    for k in range(steps):
        alpha = (k + 1) / steps
        interp = current + (targets - current) * alpha
        for i in range(6):
            p.resetJointState(tb6, i, float(interp[i]))
        p.stepSimulation()
        sync_rq(tb6, fi, rq)
    for i in range(6):
        p.resetJointState(tb6, i, float(targets[i]))
    sync_rq(tb6, fi, rq)

def pause(tb6, fi, rq, targets, msg, sec=1.0):
    print(f"  ... {msg}")
    d = time.time() + sec
    while time.time() < d:
        for i in range(6):
            p.resetJointState(tb6, i, float(targets[i]))
        p.stepSimulation()
        sync_rq(tb6, fi, rq)
        time.sleep(1.0/240.0)

def status(tb6, fi):
    j = [round(p.getJointState(tb6,i)[0],3) for i in range(6)]
    t = np.round(p.getLinkState(tb6, fi)[0], 3).tolist()
    rp = np.round(p.getLinkState(tb6, fi)[0], 3).tolist()
    print(f"  J: {j}   TCP: {t}")

# =====================================================================
def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.setTimeStep(1.0/240.0)
    p.resetDebugVisualizerCamera(1.8, 50, -20, [0.45,0.10,0.15])

    print("Loading TB6 ...")
    tb6 = load_tb6()
    fi = p.getNumJoints(tb6)-1

    print("Loading Robotiq (with constraint) ...")
    rq, jidx = load_robotiq()
    sync_rq(tb6, fi, rq)

    p.loadURDF("plane.urdf", [0,0,-0.2])
    make_platform()
    cube_id = make_cube()

    # ---- IK helper ----
    def ik_for_palm(target_pos, seed=HOME):
        """Compute joint angles to place palm at target_pos (fingers down)."""
        palm_orn = p.getQuaternionFromEuler([0, np.pi, 0])   # fingers down
        inv_mq = p.getQuaternionFromEuler([-MOUNT_RPY[0],-MOUNT_RPY[1],-MOUNT_RPY[2]])
        inv_mt = [-MOUNT_TRANS[0],-MOUNT_TRANS[1],-MOUNT_TRANS[2]]
        fl_pos, fl_orn = p.multiplyTransforms(target_pos, palm_orn, inv_mt, inv_mq)
        return np.array(p.calculateInverseKinematics(
            tb6, fi, fl_pos, fl_orn,
            lowerLimits=[-3.14]*6, upperLimits=[3.14]*6,
            jointRanges=[6.28]*6, restPoses=seed.tolist(),
            maxNumIterations=500, residualThreshold=1e-5))

    # 2-stage path:  HOME → above cube → at cube
    # This ensures the arm approaches from above, never sweeping through the table.
    above_j = ik_for_palm([PICK_XY[0], PICK_XY[1], CUBE_Z + 0.30], HOME)
    grasp_j = ik_for_palm([PICK_XY[0], PICK_XY[1], CUBE_Z + 0.06], above_j)

    print(f"  IK above:  {np.round(above_j,3).tolist()}")
    print(f"  IK grasp:  {np.round(grasp_j,3).tolist()}")
    print("="*50)

    # ---- 1: move to above cube ----
    print("\n[1] Approach from above ...")
    move_arm(tb6, fi, rq, above_j, duration=1.5)
    status(tb6, fi)
    pause(tb6, fi, rq, above_j, "above cube", 1.0)

    # ---- 2: descend to grasp ----
    print("[2] Descend to grasp ...")
    move_arm(tb6, fi, rq, grasp_j, duration=1.0)
    status(tb6, fi)
    pause(tb6, fi, rq, grasp_j, "at cube", 0.8)

    # ---- 3: close + attach cube ----
    print("[3] Grasp ...")
    set_gripper(rq, jidx, 'close')
    for _ in range(150):
        p.stepSimulation(); sync_rq(tb6, fi, rq)
    rp_pos, rp_orn = p.getBasePositionAndOrientation(rq)
    cube_in_palm = [0.0, 0.065, 0.0]
    cw_pos, _ = p.multiplyTransforms(rp_pos, rp_orn, cube_in_palm, [0,0,0,1])
    p.resetBasePositionAndOrientation(cube_id, cw_pos, [0,0,0,1])
    grasp_cid = p.createConstraint(rq, -1, cube_id, -1, p.JOINT_FIXED,
                                   [0,0,0], [0,0,0], cube_in_palm,[0,0,0,1],[0,0,0,1])
    pause(tb6, fi, rq, grasp_j, "GRASPED", 1.5)

    # ---- 4: lift back to above ----
    print("\n[4] Lift ...")
    move_arm(tb6, fi, rq, above_j, duration=1.0)
    pause(tb6, fi, rq, above_j, "lifted", 1.0)

    # ---- 5: move aside ----
    print("[5] Move aside ...")
    aside_j = ik_for_palm([PICK_XY[0]+0.15, PICK_XY[1]+0.25, CUBE_Z + 0.20], above_j)
    move_arm(tb6, fi, rq, aside_j, duration=1.5)
    pause(tb6, fi, rq, aside_j, "above drop", 1.0)

    # ---- 6: release ----
    print("[6] Release ...")
    if grasp_cid is not None:
        p.removeConstraint(grasp_cid); grasp_cid = None
    set_gripper(rq, jidx, 'open')
    pause(tb6, fi, rq, aside_j, "DROPPED", 1.5)

    # ---- 7: home ----
    print("[7] Home ...")
    move_arm(tb6, fi, rq, HOME, duration=1.5)
    status(tb6, fi)
    status(tb6, fi)

    print("\nDone.  Close window to exit.")
    try:
        while p.isConnected(): time.sleep(0.1)
    except: pass
    finally:
        try: p.disconnect()
        except: pass

if __name__ == '__main__':
    main()
