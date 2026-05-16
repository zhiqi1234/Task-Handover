"""
Active Contact Sensing for Robust Robot-to-Human Object Handover
Li, Shao & Hsu (2026) — PyBullet simulation

TB6 R5 arm + Robotiq 3F gripper + human hand proxy.
Analytical FT sensor + Bayesian contact model + firm-grasp detection.

Simulation ↔ Real-machine command mapping:
  ========================  =============================  ===================================
  Simulation                 Real TB6 R5                   Real Robotiq 3F (Modbus RTU)
  ========================  =============================  ===================================
  _hold_joints(targets)      CSP servo hold (default)      —
  _start_move(target, ...)   MoveAbsJ --jointtarget_var    —
  _step_move (smoothstep)    (built-in trajectory)         —
  probing oscillation        SpeedJ --vel={...}             —
  emergency stop freeze      Stop → Disable                 —
  open_gripper_urdf          —                             rPRA=0x00, rGTO=1
  close_gripper_urdf         —                             rPRA=0xFF, rGTO=1
  FT sensor read             topic订阅 system_rtstate      —
  FT sensor tare             —                             记录当前FT值作为零点偏移
  ========================  =============================  ===================================

TB6 initialization sequence (MUST follow this order):
  Clear → Disable → Mode(8=CSP) → SetMaxToq → Recover → SetRate(10-30) → Enable

Robotiq initialization sequence:
  Activate(rACT=1) → wait gIMC==3 → Set mode → ready for commands

GRAVITY is set to 0 — ALL bodies (TB6, Robotiq, object, hand) are positioned
kinematically via resetJointState / resetBasePositionAndOrientation. None of
them need gravity; gravity only created jitter by pulling the 46.8 kg Link2
against the kinematic reset every physics step. TB6 joints use pure kinematic
resetJointState with correct smoothstep velocity for smooth motion.
Robotiq gripper uses setJointMotorControl2 (near-zero mass, no jitter risk).

EXPERIMENT FLOW (sequential state machine):
  HOME → MOVING_TO_POSE → READY → GRASPING → HOLDING → DETECTING
       → FIRM_DETECTED → RELEASING → DONE → MOVING_TO_HOME → HOME

KEYS:
  ENTER/SPACE — advance to next phase
  g           — simulate FIRM GRASP by human (during DETECTING)
  t           — simulate incidental TOUCH (during DETECTING)
  n           — simulate NO contact (during DETECTING)
  p           — toggle probing ON/OFF (during DETECTING)
  r           — reset Bayesian model
  s           — EMERGENCY STOP (freeze arm)
  ESC         — quit
"""

import pybullet as p
import pybullet_data
import os, sys, json
import numpy as np
from collections import deque
from datetime import datetime
from enum import Enum
from scipy.stats import chi2

# =====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(BASE_DIR, "simulation")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
URDF_TB6 = os.path.join(SIM_DIR, "tb6.urdf")
URDF_ROBOTIQ = os.path.join(SIM_DIR, "robotiq_3f.urdf")
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
SIM_FREQ   = 240
GRAVITY    = 0.0           # disabled — all bodies are kinematic; gravity only causes jitter
V_MAX      = 0.04          # max gripper velocity (m/s)
OBJECT_WEIGHT = 3.0        # object weight (N)
BETA       = 0.05          # measurement noise variance
CONFIDENCE_C = 0.99        # confidence level for firm-grasp check
FT_NOISE   = 0.03          # FT sensor noise std

# ---------------------------------------------------------------------------
# Probing parameters
# ---------------------------------------------------------------------------
PROBE_AMP   = 0.012         # vertical probing amplitude (rad)
PROBE_FREQ  = 1.5           # probing frequency (Hz)
PROBE_DELTA_Z = 0.015       # vertical displacement for probing (m)

# ---------------------------------------------------------------------------
# Joint configurations
# ---------------------------------------------------------------------------
# HOME: arm folded in a compact, safe pose with TCP pointing downward.
# All joints at 0 = robot standard reference pose (straight up).
# J1=0 keeps the base stationary — real robot should NOT rotate base.
HOME_JOINTS      = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# HANDOVER: arm reaching forward to handover workspace.
# J1 is kept at 0 to match the real robot's fixed base orientation.
# *** ADJUST THESE VALUES TO MATCH YOUR ACTUAL SETUP ***
# J2/J3 control reach distance and height; J4/J5/J6 control wrist orientation.
HANDOVER_JOINTS  = np.array([0.0, -0.901, 1.886, -2.149, 0.354, 3.133])
MOVE_DURATION    = 5.0       # slow, ~v30 equivalent (seconds)

# ---------------------------------------------------------------------------
# Robotiq 3F → TB6 flange mounting
# ---------------------------------------------------------------------------
MOUNT_TRANSLATION = [0.0, 0.0, 0.048]
MOUNT_EULER       = [np.pi / 2, 0.0, 0.0]

# ---------------------------------------------------------------------------
# Hand proxy & held object offsets (in Robotiq PALM frame)
# ---------------------------------------------------------------------------
OBJECT_OFFSET = [0.0, 0.065, 0.0]
HAND_OFFSET  = [0.0, 0.065, 0.0]


# =====================================================================
# Experiment State Machine
# =====================================================================
class ExpState(Enum):
    HOME            = 0
    MOVING_TO_POSE  = 1
    READY           = 2
    GRASPING        = 3
    HOLDING         = 4
    DETECTING       = 5
    FIRM_DETECTED   = 6
    RELEASING        = 7
    DONE            = 8
    MOVING_TO_HOME  = 9
    EMERGENCY_STOP  = 10


# =====================================================================
class AnalyticalFTSensor:
    """Simulated wrist FT sensor — returns 6-DOF wrench."""

    def __init__(self):
        self.contact_type = 'none'
        self.damping = 0.3
        self.k = {
            'none':  (0.0, 0.0),
            'touch': (5.0, 0.5),
            'grasp': (80.0, 60.0),
        }

    def set_contact(self, c):
        self.contact_type = c

    def read(self, u_z):
        ku, kd = self.k[self.contact_type]
        f = ku * max(u_z, 0) + kd * (-min(u_z, 0)) - self.damping * u_z
        f += np.random.normal(0, FT_NOISE)
        ft = np.zeros(6)
        ft[2] = f
        ft[0] = np.random.normal(0, FT_NOISE * 0.15)
        ft[1] = np.random.normal(0, FT_NOISE * 0.15)
        return ft


# =====================================================================
class RealFTSensor:
    """Real TB6 wrist FT sensor via topic subscription.
    Call tare() after gripper grasps object, before human contact.
    """

    def __init__(self, ip="192.168.50.1", port=19091):
        import topic, message, threading
        self.topic = topic
        self.message = message
        self.lock = threading.Lock()
        self._raw = np.zeros(6)
        self._tare = np.zeros(6)
        self._tared = False

        options = topic.NodeOptions()
        options.node_name = 'ft_reader'
        options.sub_url = f'tcp://{ip}:{port}'
        self._node = topic.Node(options)
        if not self._node.Start():
            raise RuntimeError("Failed to start topic node")
        self._sub = self._node.CreateSubscriptionRT(
            "system_rtstate", self._on_rtstate)
        self._running = True

    def _on_rtstate(self, tt):
        parm = self.message.SystemStateData()
        self.message.display_rt(tt, parm)
        if parm.controller.ftvalues:
            ft = parm.controller.ftvalues[0]
            with self.lock:
                self._raw = np.array([ft.fx, ft.fy, ft.fz,
                                      ft.mx, ft.my, ft.mz])

    def tare(self):
        with self.lock:
            self._tare = self._raw.copy()
            self._tared = True
        print(f"[FT] Tare: {np.round(self._tare[:3], 3).tolist()} N")

    def read(self, u_z=None):
        with self.lock:
            raw = self._raw.copy()
        return raw - self._tare if self._tared else raw

    def close(self):
        self._running = False
        self._node.Shutdown()

    @property
    def fz(self):
        return self.read()[2]


# =====================================================================
class BayesianContactModel:
    """Piecewise-linear Bayesian model of human-object contact state."""

    def __init__(self):
        self.beta = BETA
        self.d = 2
        self.m = np.zeros(2)
        self.S = np.eye(2) * 100.0
        self.S_inv = np.eye(2) / 100.0
        self.data = deque(maxlen=200)

    def reset(self):
        self.__init__()

    def phi(self, u):
        u = float(u)
        return np.array([max(u, 0.0), -min(u, 0.0)])

    def update(self, u, f):
        u, f = float(u), float(f)
        self.data.append((u, f))
        ph = self.phi(u)
        self.S_inv += (1.0 / self.beta) * np.outer(ph, ph)
        self.S = np.linalg.inv(self.S_inv)
        self.m = self.S @ (self.S_inv @ self.m + (1.0 / self.beta) * f * ph)

    def entropy(self):
        _, ld = np.linalg.slogdet(self.S)
        return 0.5 * self.d * (1 + np.log(2 * np.pi)) + 0.5 * ld

    def check_firm_grasp(self, ow=OBJECT_WEIGHT, conf=CONFIDENCE_C):
        for ft in [0.5 * ow, 0.5, -0.5]:
            try:
                P = np.linalg.cholesky(self.S * chi2.ppf(conf, df=2))
            except Exception:
                return False
            E = np.array([1.0, 0.0])
            w_lo = max(0.0, E @ self.m - np.linalg.norm(E @ P))
            if 0 <= ft <= w_lo * V_MAX:
                continue
            E = np.array([0.0, 1.0])
            w_hi = E @ self.m + np.linalg.norm(E @ P)
            if -w_hi * V_MAX <= ft <= 0:
                continue
            return False
        return True


# =====================================================================
# Robotiq 3F helpers  (setJointMotorControl2 is fine here — near-zero mass)
# =====================================================================
_FINGER_BASE_JOINTS = ['palm_finger_1_joint', 'palm_finger_2_joint']
_FINGER_KNUCKLE_JOINTS = [
    'finger_1_joint_1', 'finger_1_joint_2', 'finger_1_joint_3',
    'finger_2_joint_1', 'finger_2_joint_2', 'finger_2_joint_3',
    'finger_middle_joint_1', 'finger_middle_joint_2', 'finger_middle_joint_3',
]


def load_robotiq_kinematic(pos, orn):
    rq = p.loadURDF(URDF_ROBOTIQ, pos, orn, useFixedBase=False)
    jidx = {}
    for i in range(p.getNumJoints(rq)):
        name = p.getJointInfo(rq, i)[1].decode()
        jidx[name] = i
        p.changeDynamics(rq, i, mass=0.001, jointDamping=0.0)
        lo, _ = p.getJointInfo(rq, i)[8:10]
        p.resetJointState(rq, i, lo)
        p.setJointMotorControl2(rq, i, p.POSITION_CONTROL,
                                targetPosition=lo, force=5.0, maxVelocity=0.8)
    p.changeDynamics(rq, -1, mass=0.001)
    return rq, jidx


def _set_all_fingers(rq, jidx, action):
    force, max_v = 5.0, 0.8
    if action == 'close':
        for key in _FINGER_BASE_JOINTS:
            if key not in jidx:
                continue
            _, hi = p.getJointInfo(rq, jidx[key])[8:10]
            p.setJointMotorControl2(rq, jidx[key], p.POSITION_CONTROL,
                                    targetPosition=hi, force=force,
                                    maxVelocity=max_v)
        for key in _FINGER_KNUCKLE_JOINTS:
            if key not in jidx:
                continue
            lo, hi = p.getJointInfo(rq, jidx[key])[8:10]
            tgt = lo + 0.8 * (hi - lo)
            p.setJointMotorControl2(rq, jidx[key], p.POSITION_CONTROL,
                                    targetPosition=tgt, force=force,
                                    maxVelocity=max_v)
    else:
        for key in _FINGER_BASE_JOINTS + _FINGER_KNUCKLE_JOINTS:
            if key not in jidx:
                continue
            lo, _ = p.getJointInfo(rq, jidx[key])[8:10]
            p.setJointMotorControl2(rq, jidx[key], p.POSITION_CONTROL,
                                    targetPosition=lo, force=force,
                                    maxVelocity=max_v)


def open_gripper_urdf(rq, jidx):
    _set_all_fingers(rq, jidx, 'open')


def close_gripper_urdf(rq, jidx):
    _set_all_fingers(rq, jidx, 'close')


# =====================================================================
def _multiply_pose(pos, orn, offset_pos, offset_orn):
    return p.multiplyTransforms(pos, orn, offset_pos, offset_orn)


# =====================================================================
class HandoverSim:
    def __init__(self):
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(1.0 / SIM_FREQ)
        p.setRealTimeSimulation(0)

        p.resetDebugVisualizerCamera(2.0, 50, -20, [0.5, 0.1, 0.2])

        self.tb6         = None
        self.robotiq     = None
        self.rq_jidx     = {}
        self.jidx        = {}
        self.flange_idx  = None

        self.object_vis  = None
        self.hand_proxy  = None

        self.ft    = AnalyticalFTSensor()
        self.model = BayesianContactModel()

        # ---- state machine ----
        self.state       = ExpState.HOME
        self.probing     = False
        self.contact     = 'none'
        self.t           = 0.0
        self.u_z         = 0.0
        self.base_joints = HOME_JOINTS.copy()
        self.log = deque(maxlen=10000)

        # ---- arm movement interpolation ----
        self._move_in_progress  = False
        self._move_start_joints = None
        self._move_target_joints = None
        self._move_duration     = 0.0
        self._move_elapsed      = 0.0
        self._move_next_state   = None

        # ---- gripper animation timers ----
        self._grasp_timer   = 0.0
        self._release_timer = 0.0

        # ---- emergency stop ----
        self._emergency_joints = None

        self.mount_orn_offset = p.getQuaternionFromEuler(MOUNT_EULER)

    # ==================================================================
    def _hold_joints(self, targets, velocities=None):
        """Set TB6 joint positions.  Corresponds to TB6 CSP servo hold.

        Pure kinematic control: resetJointState with correct velocity.
        Gravity is disabled globally (setGravity(0,0,0)) because ALL bodies
        in this simulation are positioned kinematically — none need gravity.
        This completely eliminates the 'kinematic reset vs. gravity drift'
        conflict that caused persistent jitter on heavy links.
        """
        if velocities is None:
            velocities = np.zeros(6)
        for i in range(6):
            p.resetJointState(self.tb6, i, float(targets[i]),
                              targetVelocity=float(velocities[i]))

    # ==================================================================
    def setup(self):
        p.loadURDF("plane.urdf", [0, 0, -0.2])

        # ---- load TB6 (fixed base) ----
        self.tb6 = p.loadURDF(URDF_TB6, [0, 0, 0], [0, 0, 0, 1],
                              useFixedBase=True)
        for i in range(p.getNumJoints(self.tb6)):
            name = p.getJointInfo(self.tb6, i)[1].decode()
            self.jidx[name] = i

        # ---- start at HOME ----
        self.flange_idx = p.getNumJoints(self.tb6) - 1
        self._hold_joints(HOME_JOINTS)
        for _ in range(200):
            p.stepSimulation()

        st = p.getLinkState(self.tb6, self.flange_idx)
        flange_pos, flange_orn = np.array(st[0]), np.array(st[1])
        print(f"TB6 HOME joints: {np.round(HOME_JOINTS,3).tolist()}")
        print(f"TB6 HOME TCP: ({flange_pos[0]:.3f}, "
              f"{flange_pos[1]:.3f}, {flange_pos[2]:.3f})")

        # ---- Robotiq 3F (uses setJointMotorControl2 — near-zero mass, OK) ----
        rq_init_pos, rq_init_orn = _multiply_pose(
            flange_pos, flange_orn,
            MOUNT_TRANSLATION, self.mount_orn_offset)
        self.robotiq, self.rq_jidx = load_robotiq_kinematic(rq_init_pos,
                                                            rq_init_orn)
        open_gripper_urdf(self.robotiq, self.rq_jidx)

        for _ in range(50):
            p.stepSimulation()

        # ---- held object visual ----
        os_ = p.createCollisionShape(p.GEOM_BOX,
                                     halfExtents=[0.015, 0.015, 0.06])
        ov_ = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[0.015, 0.015, 0.06],
                                  rgbaColor=[0.95, 0.25, 0.15, 1])
        self.object_vis = p.createMultiBody(0.001, os_, ov_,
                                            [0, 0, 0], [0, 0, 0, 1])

        # ---- human hand proxy ----
        hs_ = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
        hv_ = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                                  rgbaColor=[0.2, 0.5, 0.9, 0.5])
        self.hand_proxy = p.createMultiBody(0.001, hs_, hv_,
                                            [0, 0, 0], [0, 0, 0, 1])

        self._sync_rq_to_flange()
        self._sync_attachments()

        for _ in range(50):
            p.stepSimulation()

        # ---- HUD ----
        self.hud = []
        for i in range(6):
            tid = p.addUserDebugText("", [0, 0, 1.3 - i * 0.08],
                                     textColorRGB=[1, 1, 1], textSize=1.8)
            self.hud.append(tid)
        self._redraw()

        print("\n" + "=" * 60)
        print("EXPERIMENT FLOW: HOME → MOVE TO POSE → GRASP OBJECT")
        print("               → HOLD → DETECT GRASP → RELEASE → HOME")
        print("=" * 60)
        print("  ENTER/SPACE — advance to next phase")
        print("  g — firm grasp | t — touch | n — no contact (in DETECTING)")
        print("  p — toggle probe | r — reset model | s — EMERGENCY STOP")
        print("  ESC — quit")
        print()

    # ==================================================================
    # Arm movement  (smoothstep interp → resetJointState each step)
    # ==================================================================
    def _get_current_joints(self):
        return np.array([p.getJointState(self.tb6, i)[0] for i in range(6)])

    def _start_move(self, target_joints, next_state, duration=None):
        if duration is None:
            duration = MOVE_DURATION
        self._move_start_joints = self._get_current_joints()
        self._move_target_joints = np.asarray(target_joints, dtype=float)
        self._move_duration = duration
        self._move_elapsed = 0.0
        self._move_in_progress = True
        self._move_next_state = next_state

    def _step_move(self):
        self._move_elapsed += 1.0 / SIM_FREQ
        t = min(self._move_elapsed / self._move_duration, 1.0)
        # smoothstep:  alpha = t²(3 - 2t)
        alpha = t * t * (3.0 - 2.0 * t)
        interp = (self._move_start_joints +
                  (self._move_target_joints - self._move_start_joints) * alpha)
        # Derivative: dα/dt = 6·t·(1-t) / duration  (0 at both ends, smooth)
        if t < 1.0:
            dalpha = 6.0 * t * (1.0 - t) / self._move_duration
        else:
            dalpha = 0.0
        velocities = (self._move_target_joints - self._move_start_joints) * dalpha
        self._hold_joints(interp, velocities)
        if t >= 1.0:
            self._move_in_progress = False
            self.state = self._move_next_state
            self.base_joints = self._move_target_joints.copy()
            return True
        return False

    # ==================================================================
    def _sync_rq_to_flange(self):
        fs = p.getLinkState(self.tb6, self.flange_idx)
        rpos, rorn = _multiply_pose(fs[0], fs[1],
                                    MOUNT_TRANSLATION, self.mount_orn_offset)
        p.resetBasePositionAndOrientation(self.robotiq, rpos, rorn)

    def _get_palm_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.robotiq)
        return np.array(pos), np.array(orn)

    def _sync_attachments(self):
        palm_pos, palm_orn = self._get_palm_pose()

        obj_pos, obj_orn = _multiply_pose(palm_pos, palm_orn,
                                          OBJECT_OFFSET, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.object_vis, obj_pos, obj_orn)

        if self.contact == 'none':
            h_off = [HAND_OFFSET[0], HAND_OFFSET[1] - 0.15, HAND_OFFSET[2]]
        elif self.contact == 'touch':
            h_off = [HAND_OFFSET[0], HAND_OFFSET[1] + 0.01, HAND_OFFSET[2] - 0.06]
        else:
            h_off = [HAND_OFFSET[0], HAND_OFFSET[1] + 0.03, HAND_OFFSET[2]]

        hand_pos, hand_orn = _multiply_pose(palm_pos, palm_orn,
                                            h_off, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.hand_proxy, hand_pos, hand_orn)

    # ==================================================================
    def _advance_state(self):
        if self.state == ExpState.HOME:
            self.state = ExpState.MOVING_TO_POSE
            self._start_move(HANDOVER_JOINTS, ExpState.READY)
            print("\n  >>> Moving to experiment pose...")

        elif self.state == ExpState.READY:
            self.state = ExpState.GRASPING
            close_gripper_urdf(self.robotiq, self.rq_jidx)
            self._grasp_timer = 0.0
            self.contact = 'grasp'
            self.ft.set_contact('grasp')
            print("\n  >>> Gripper closing to grasp object...")

        elif self.state == ExpState.HOLDING:
            self.state = ExpState.DETECTING
            self.probing = True
            self.model.reset()
            self.contact = 'none'
            self.ft.set_contact('none')
            print("\n  >>> Detection started — probing ON, model reset.")
            print("      Press 'g' to simulate human firm grasp,")
            print("      or wait for automatic detection.")

        elif self.state == ExpState.FIRM_DETECTED:
            self.state = ExpState.RELEASING
            self.probing = False
            self.u_z = 0.0
            open_gripper_urdf(self.robotiq, self.rq_jidx)
            self._release_timer = 0.0
            print("\n  >>> Firm grasp confirmed! Releasing object...")

        elif self.state == ExpState.DONE:
            self.state = ExpState.MOVING_TO_HOME
            self._start_move(HOME_JOINTS, ExpState.HOME)
            print("\n  >>> Returning to home position...")

        elif self.state == ExpState.EMERGENCY_STOP:
            self.state = ExpState.MOVING_TO_HOME
            self._start_move(HOME_JOINTS, ExpState.HOME)
            print("\n  >>> Emergency released. Returning to home...")

    # ==================================================================
    def _redraw(self):
        state_names = {
            ExpState.HOME:            "HOME",
            ExpState.MOVING_TO_POSE:  "MOVING TO POSE...",
            ExpState.READY:           "READY (at handover pose)",
            ExpState.GRASPING:        "GRASPING...",
            ExpState.HOLDING:         "HOLDING object",
            ExpState.DETECTING:       "DETECTING grasp...",
            ExpState.FIRM_DETECTED:   "*** FIRM GRASP! ***",
            ExpState.RELEASING:        "RELEASING object...",
            ExpState.DONE:            "DONE — experiment complete",
            ExpState.MOVING_TO_HOME:  "RETURNING HOME...",
            ExpState.EMERGENCY_STOP:  "!!! EMERGENCY STOP !!!",
        }

        firm = self.model.check_firm_grasp()
        st = state_names.get(self.state, str(self.state))

        lines = [
            f"Phase: {st}",
            f"Probing: {'ON' if self.probing else 'OFF'}    "
            f"Contact: {self.contact.upper()}",
            f"Firm Grasp: {'YES!' if firm else 'no'}    "
            f"H(w)={self.model.entropy():.1f}  w=[{self.model.m[0]:.0f},{self.model.m[1]:.0f}]",
        ]

        if self.state == ExpState.HOME:
            lines.append("Press ENTER to begin experiment")
        elif self.state == ExpState.READY:
            lines.append("Press ENTER to grasp object")
        elif self.state == ExpState.HOLDING:
            lines.append("Press ENTER to start detection")
        elif self.state == ExpState.DETECTING:
            lines.append("Wait for detection OR press 'g' (grasp) / 't' (touch) / 'n' (none)")
        elif self.state == ExpState.FIRM_DETECTED:
            lines.append("Press ENTER to release object")
        elif self.state == ExpState.DONE:
            lines.append("Press ENTER to return home, or ESC to quit")
        elif self.state == ExpState.EMERGENCY_STOP:
            lines.append("Press ENTER to recover and return home")
        else:
            lines.append("")

        lines.append("ENTER=advance  g/t/n=contact  p=probe  r=reset  s=ESTOP  ESC=quit")

        colors = [
            [0.3, 1, 0.3] if self.state == ExpState.FIRM_DETECTED else
            [1, 0.8, 0.3] if self.state == ExpState.DETECTING else
            [1, 0.3, 0.3] if self.state == ExpState.EMERGENCY_STOP else
            [0.7, 0.7, 0.7],
            [0.2, 1, 0.2] if firm else [0.7, 0.7, 0.7],
            [1, 1, 0.7],
            [0.5, 0.8, 1],
            [0.5, 0.5, 0.5] if self.state != ExpState.EMERGENCY_STOP else [1, 0.3, 0.3],
            [0.4, 0.6, 0.8],
        ]

        for i, (txt, color) in enumerate(zip(lines, colors)):
            try:
                p.addUserDebugText(txt, [0, 0, 1.3 - i * 0.08],
                                   textColorRGB=color, textSize=1.8,
                                   replaceItemUniqueId=self.hud[i])
            except p.error:
                pass

    # ==================================================================
    def step(self):
        # ---- Emergency stop ----
        if self.state == ExpState.EMERGENCY_STOP:
            self._hold_joints(self._emergency_joints)
            p.stepSimulation()
            self._sync_rq_to_flange()
            self._sync_attachments()
            self.t += 1.0 / SIM_FREQ
            return {'t': self.t, 'u_z': 0.0, 'f_z': 0.0,
                    'contact': 'none', 'probing': False,
                    'firm': False, 'entropy': self.model.entropy(),
                    'w_up': float(self.model.m[0]),
                    'w_down': float(self.model.m[1]),
                    'state': self.state.name}

        # ---- Arm movement in progress ----
        if self._move_in_progress:
            done = self._step_move()
            p.stepSimulation()
            self._sync_rq_to_flange()
            self._sync_attachments()
            self.t += 1.0 / SIM_FREQ
            if done:
                if self.state == ExpState.READY:
                    print("  >>> Arm at experiment pose. Press ENTER to grasp.")
                elif self.state == ExpState.HOME:
                    print("  >>> Arm back at home position.")
            return {'t': self.t, 'u_z': 0.0, 'f_z': 0.0,
                    'contact': self.contact, 'probing': False,
                    'firm': False, 'entropy': self.model.entropy(),
                    'w_up': float(self.model.m[0]),
                    'w_down': float(self.model.m[1]),
                    'state': self.state.name}

        # ---- Gripper animation: GRASPING ----
        if self.state == ExpState.GRASPING:
            self._grasp_timer += 1.0 / SIM_FREQ
            self._hold_joints(self.base_joints)
            p.stepSimulation()
            self._sync_rq_to_flange()
            self._sync_attachments()
            self.t += 1.0 / SIM_FREQ
            if self._grasp_timer >= 1.2:
                self.state = ExpState.HOLDING
                print("  >>> Object grasped. Press ENTER to start detection.")
            return {'t': self.t, 'u_z': 0.0, 'f_z': 0.0,
                    'contact': self.contact, 'probing': False,
                    'firm': False, 'entropy': self.model.entropy(),
                    'w_up': float(self.model.m[0]),
                    'w_down': float(self.model.m[1]),
                    'state': self.state.name}

        # ---- Gripper animation: RELEASING ----
        if self.state == ExpState.RELEASING:
            self._release_timer += 1.0 / SIM_FREQ
            self._hold_joints(self.base_joints)
            p.stepSimulation()
            self._sync_rq_to_flange()
            self._sync_attachments()
            self.t += 1.0 / SIM_FREQ
            if self._release_timer >= 1.2:
                self.state = ExpState.DONE
                print("  >>> Object released! Experiment complete.")
                print("      Press ENTER to return home, or ESC to quit.")
            return {'t': self.t, 'u_z': 0.0, 'f_z': 0.0,
                    'contact': 'none', 'probing': False,
                    'firm': False, 'entropy': self.model.entropy(),
                    'w_up': float(self.model.m[0]),
                    'w_down': float(self.model.m[1]),
                    'state': self.state.name}

        # ----
        # Steady states: HOME, READY, HOLDING, DETECTING, FIRM_DETECTED, DONE
        # ----

        # Compute joint targets (and velocities for smooth rendering)
        if self.probing and self.state == ExpState.DETECTING:
            omega = 2 * np.pi * PROBE_FREQ
            phase = np.sin(omega * self.t)
            dphase = omega * np.cos(omega * self.t)
            targets = self.base_joints.copy()
            velocities = np.zeros(6)
            targets[1] += PROBE_AMP * phase
            targets[2] += PROBE_AMP * 0.6 * phase
            targets[4] -= PROBE_AMP * 0.3 * phase
            velocities[1] = PROBE_AMP * dphase
            velocities[2] = PROBE_AMP * 0.6 * dphase
            velocities[4] = -PROBE_AMP * 0.3 * dphase
            self.u_z = np.clip(PROBE_DELTA_Z * dphase, -V_MAX, V_MAX)
        else:
            targets = self.base_joints
            velocities = None
            self.u_z = 0.0

        self._hold_joints(targets, velocities)
        p.stepSimulation()
        self._sync_rq_to_flange()
        self._sync_attachments()

        # FT sensor + Bayesian model
        if self.state == ExpState.DETECTING:
            ft = self.ft.read(self.u_z)
            self.model.update(self.u_z, ft[2])

            if self.model.check_firm_grasp():
                self.state = ExpState.FIRM_DETECTED
                self.probing = False
                self.u_z = 0.0
                self.contact = 'grasp'
                self.ft.set_contact('grasp')
                print("\n  >>> FIRM GRASP DETECTED AUTOMATICALLY! <<<")
                print("      Press ENTER to release object.")
        else:
            ft = self.ft.read(0.0)

        self.t += 1.0 / SIM_FREQ

        r = {'t': self.t, 'u_z': self.u_z, 'f_z': ft[2],
             'contact': self.contact, 'probing': self.probing,
             'firm': self.state == ExpState.FIRM_DETECTED,
             'entropy': self.model.entropy(),
             'w_up': float(self.model.m[0]),
             'w_down': float(self.model.m[1]),
             'state': self.state.name}
        self.log.append(r)
        return r

    # ==================================================================
    def run(self):
        print("\nPress ENTER to begin the experiment.\n")
        report = SIM_FREQ
        i = 0
        try:
            while True:
                try:
                    keys = p.getKeyboardEvents()
                except p.error:
                    break

                for k, v in keys.items():
                    if v & p.KEY_WAS_TRIGGERED:
                        # --- ENTER / SPACE: advance state ---
                        if k in (13, 32):
                            if self.state == ExpState.EMERGENCY_STOP:
                                self._advance_state()
                            elif self.state not in (
                                    ExpState.MOVING_TO_POSE,
                                    ExpState.GRASPING,
                                    ExpState.RELEASING,
                                    ExpState.MOVING_TO_HOME,
                                    ExpState.DETECTING):
                                self._advance_state()

                        # --- s: emergency stop ---
                        elif k == ord('s'):
                            if self.state != ExpState.EMERGENCY_STOP:
                                self.state = ExpState.EMERGENCY_STOP
                                self._emergency_joints = self._get_current_joints()
                                self._move_in_progress = False
                                self.probing = False
                                self.u_z = 0.0
                                print("\n  !!! EMERGENCY STOP !!!")
                                print("      Arm frozen at current position.")
                                print("      Press ENTER to recover and return home.")

                        # --- p: toggle probing ---
                        elif k == ord('p'):
                            if self.state == ExpState.DETECTING:
                                self.probing = not self.probing
                                print(f"  PROBE: {'ON' if self.probing else 'OFF'}")

                        # --- g / t / n: contact type (only in DETECTING) ---
                        elif k == ord('g'):
                            if self.state == ExpState.DETECTING:
                                self.contact = 'grasp'
                                self.ft.set_contact('grasp')
                                print("  CONTACT: FIRM GRASP (human)")
                        elif k == ord('t'):
                            if self.state == ExpState.DETECTING:
                                self.contact = 'touch'
                                self.ft.set_contact('touch')
                                print("  CONTACT: INCIDENTAL TOUCH")
                        elif k == ord('n'):
                            if self.state == ExpState.DETECTING:
                                self.contact = 'none'
                                self.ft.set_contact('none')
                                print("  CONTACT: NONE")

                        # --- r: reset model ---
                        elif k == ord('r'):
                            self.model.reset()
                            print("  MODEL RESET")

                        # --- o / c: manual gripper ---
                        elif k == ord('o'):
                            open_gripper_urdf(self.robotiq, self.rq_jidx)
                            print("  GRIPPER: OPEN (manual)")
                        elif k == ord('c'):
                            close_gripper_urdf(self.robotiq, self.rq_jidx)
                            print("  GRIPPER: CLOSE (manual)")

                        # --- ESC: quit ---
                        elif k == 27:
                            print("Quit.")
                            self._save_log()
                            p.disconnect()
                            return

                r = self.step()
                if r is None:
                    break

                if i % (SIM_FREQ // 2) == 0:
                    self._redraw()

                if i % report == 0:
                    state_str = f"{r['state']:20s}"
                    firm_str = '*** FIRM! ***' if r['firm'] else '-'
                    if self.state in (ExpState.DETECTING, ExpState.FIRM_DETECTED):
                        print(f"t={r['t']:5.1f}s | {state_str} | "
                              f"u={r['u_z']:+5.2f} fz={r['f_z']:+6.2f}N | "
                              f"firm={firm_str:14s} "
                              f"w=[{r['w_up']:5.0f},{r['w_down']:5.0f}] "
                              f"ent={r['entropy']:.1f}")
                    elif self.state not in (ExpState.HOME, ExpState.READY):
                        print(f"t={r['t']:5.1f}s | {state_str}")

                i += 1

        except (KeyboardInterrupt, p.error):
            pass
        finally:
            try:
                self._save_log()
                p.disconnect()
            except p.error:
                pass

    def _save_log(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS_DIR, f"handover_log_{ts}.json")
        try:
            with open(path, 'w') as f:
                json.dump([{k: (float(v) if isinstance(v, (np.floating,
                               float, np.integer, int))
                                else v.tolist() if isinstance(v, np.ndarray)
                                else v)
                            for k, v in r.items()}
                           for r in self.log], f, indent=2)
            print(f"Log: {path}")
        except Exception:
            pass


# =====================================================================
def run_tests():
    print("\n" + "=" * 60 + "\nBayesian Model Unit Tests\n" + "=" * 60)

    print("\n[1] Firm grasp + bidirectional probing")
    m = BayesianContactModel()
    for i in range(300):
        u = 0.04 * np.sin(2 * np.pi * 2.0 * i / SIM_FREQ)
        f = 80 * max(u, 0) + 60 * (-min(u, 0)) - 0.3 * u + np.random.normal(0, 0.03)
        m.update(u, f)
    assert m.check_firm_grasp(), "FAIL 1"
    print(f"    w=[{m.m[0]:.0f},{m.m[1]:.0f}] firm=True  PASS")

    print("\n[2] Light touch + bidirectional probing")
    m = BayesianContactModel()
    for i in range(300):
        u = 0.04 * np.sin(2 * np.pi * 2.0 * i / SIM_FREQ)
        f = 5 * max(u, 0) + 0.5 * (-min(u, 0)) - 0.3 * u + np.random.normal(0, 0.03)
        m.update(u, f)
    assert not m.check_firm_grasp(), "FAIL 2"
    print(f"    w=[{m.m[0]:.0f},{m.m[1]:.0f}] firm=False  PASS")

    print("\n[3] Firm grasp + passive (downward only)")
    m = BayesianContactModel()
    for i in range(300):
        u = -0.03 + np.random.normal(0, 0.003)
        f = 80 * max(u, 0) + 60 * (-min(u, 0)) - 0.3 * u + np.random.normal(0, 0.03)
        m.update(u, f)
    assert m.entropy() > 0.5, "FAIL 3"
    print(f"    w=[{m.m[0]:.0f},{m.m[1]:.0f}] ent={m.entropy():.1f} (high)  PASS")
    print("\n" + "=" * 60 + "\nALL TESTS PASSED\n" + "=" * 60)


# =====================================================================
if __name__ == '__main__':
    if '--test' in sys.argv:
        run_tests()
    else:
        run_tests()
        sim = HandoverSim()
        sim.setup()
        sim.run()
