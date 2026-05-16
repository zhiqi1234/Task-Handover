"""
Active Contact Sensing for Robust Robot-to-Human Object Handover
Li, Shao & Hsu (2026) — PyBullet simulation

TB6 R5 arm + Robotiq 3F gripper + human hand proxy.
Analytical FT sensor + Bayesian contact model + firm-grasp detection.

KEYS:
  p       — toggle probing ON/OFF  (starts ON)
  g       — FIRM GRASP (human grips object firmly)
  t       — incidental TOUCH (human lightly touches object)
  n       — NO contact (human moves away)
  r       — reset Bayesian model
  o / c   — open / close gripper (visual only)
  ESC     — quit

The HUD shows: contact type, probing state, firm-grasp detection result,
and the Bayesian model's weight estimates w=[w_up, w_down].
"""

import pybullet as p
import pybullet_data
import os, sys, json
import numpy as np
from collections import deque
from datetime import datetime
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
GRAVITY    = -9.81
V_MAX      = 0.04          # max gripper velocity  (m/s)
OBJECT_WEIGHT = 3.0        # object weight (N)
BETA       = 0.05          # measurement noise variance
CONFIDENCE_C = 0.99        # confidence level for firm-grasp check
FT_NOISE   = 0.03          # FT sensor noise std

# ---------------------------------------------------------------------------
# Probing parameters
# ---------------------------------------------------------------------------
PROBE_AMP   = 0.012         # vertical probing amplitude (rad, approx)
PROBE_FREQ  = 1.5           # probing frequency (Hz)
PROBE_DELTA_Z = 0.015       # vertical displacement for probing (m)

# ---------------------------------------------------------------------------
# Robotiq 3F → TB6 flange mounting
# ---------------------------------------------------------------------------
# Palm mesh extents (metres):
#   X: [-0.070, +0.063]   Y: [-0.054, +0.051]   Z: [-0.065, +0.065]
# Mounting surface = palm -Y face (Y ≈ -0.054)
# Fingers extend from +Y side (Y ≈ +0.051)
#
# MOUNT_EULER:  rotation from PALM frame to FLANGE frame
#   Rx(+π/2): (X→X, Y→+Z, Z→-Y)
#   →  palm +Y (fingers) → flange +Z (outward from arm)  ✓
#   →  palm -Y (mount surface) → flange -Z (toward arm)
#
# MOUNT_TRANSLATION:  palm origin in flange frame
#   Palm Y=-0.054 (mount surface) → flange Z = -0.054 from palm origin
#   For mount surface AT flange origin: +0.054
#   Slightly less to hide root inside flange: +0.048
#
MOUNT_TRANSLATION = [0.0, 0.0, 0.048]
MOUNT_EULER       = [np.pi / 2, 0.0, 0.0]

# ---------------------------------------------------------------------------
# Hand proxy & held object offsets (in Robotiq PALM frame)
# ---------------------------------------------------------------------------
# Palm frame:  X=±0.07 finger spread,  Y≈+0.05 finger tips,  Z=±0.06
# Object sits at centre of grasp (between 3 fingers when closed)
# Hand proxy appears near the object
OBJECT_OFFSET = [0.0, 0.065, 0.0]     # in front of palm, between fingers
HAND_OFFSET  = [0.0, 0.065, 0.0]      # same region as object


# =====================================================================
class AnalyticalFTSensor:
    """Simulated wrist FT sensor – returns 6-DOF wrench.
    Uses a simple spring-damper model driven by desired velocity u_z
    and the current contact_type (none/touch/grasp).
    """
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
        """u_z = vertical desired velocity (m/s).  Returns 6-DOF wrench."""
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

    The TB6's built-in 6-axis FT sensor outputs fx,fy,fz (N) and mx,my,mz (Nm)
    via system_rtstate.ftvalues.  These readings INCLUDE:
      - tool / gripper / object weight (gravity component)
      - external forces from human contact

    To extract the human-contact force, we record a TARE value when nothing
    is touching the object, then subtract it from subsequent readings.
    This is a simplified alternative to the momentum observer in the paper.
    """
    def __init__(self, ip="192.168.50.1", port=19091):
        import topic, message, threading
        self.topic = topic
        self.message = message
        self.lock = threading.Lock()
        self._raw = np.zeros(6)   # [fx, fy, fz, mx, my, mz]
        self._tare = np.zeros(6)
        self._tared = False

        # start topic subscription in background
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
        """Record current reading as baseline (call when no human contact)."""
        with self.lock:
            self._tare = self._raw.copy()
            self._tared = True
        print(f"[FT] Tare: {np.round(self._tare[:3], 3).tolist()} N")

    def read(self, u_z=None):
        """Return external force (6-DOF). u_z is ignored (real sensor)."""
        with self.lock:
            raw = self._raw.copy()
        if self._tared:
            ext = raw - self._tare
        else:
            ext = raw
        return ext   # [fx, fy, fz, mx, my, mz]

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
        """Return True if the model predicts the human can support weight
        AND apply opposing forces (definition of firm grasp)."""
        for ft in [0.5 * ow, 0.5, -0.5]:
            try:
                P = np.linalg.cholesky(self.S * chi2.ppf(conf, df=2))
            except Exception:
                return False
            # Check upward segment: feasible if 0 <= ft <= w_lo * vmax
            E = np.array([1.0, 0.0])
            w_lo = max(0.0, E @ self.m - np.linalg.norm(E @ P))
            if 0 <= ft <= w_lo * V_MAX:
                continue
            # Check downward segment: feasible if -w_hi * vmax <= ft <= 0
            E = np.array([0.0, 1.0])
            w_hi = E @ self.m + np.linalg.norm(E @ P)
            if -w_hi * V_MAX <= ft <= 0:
                continue
            return False
        return True


# =====================================================================
# Robotiq 3F helper functions
# =====================================================================
_FINGER_BASE_JOINTS = ['palm_finger_1_joint', 'palm_finger_2_joint']
_FINGER_KNUCKLE_JOINTS = [
    'finger_1_joint_1', 'finger_1_joint_2', 'finger_1_joint_3',
    'finger_2_joint_1', 'finger_2_joint_2', 'finger_2_joint_3',
    'finger_middle_joint_1', 'finger_middle_joint_2', 'finger_middle_joint_3',
]


def load_robotiq_kinematic(pos, orn):
    """Load Robotiq 3F URDF with negligible mass (kinematic-only).
    Returns (body_id, joint_index_dict).
    """
    rq = p.loadURDF(URDF_ROBOTIQ, pos, orn, useFixedBase=False)
    jidx = {}
    for i in range(p.getNumJoints(rq)):
        name = p.getJointInfo(rq, i)[1].decode()
        jidx[name] = i
        p.changeDynamics(rq, i, mass=0.001, jointDamping=0.0)
        # initialise to lower limit (open)
        lo, _ = p.getJointInfo(rq, i)[8:10]
        p.resetJointState(rq, i, lo)
        p.setJointMotorControl2(rq, i, p.POSITION_CONTROL,
                                targetPosition=lo, force=5.0, maxVelocity=0.8)

    p.changeDynamics(rq, -1, mass=0.001)
    return rq, jidx


def _set_all_fingers(rq, jidx, action):
    """Open or close ALL finger joints (base + knuckles)."""
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
    """Combine a world pose with a local offset.  Returns (world_pos, world_orn)."""
    return p.multiplyTransforms(pos, orn, offset_pos, offset_orn)


# =====================================================================
class HandoverSim:
    def __init__(self):
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(1.0 / SIM_FREQ)
        p.setRealTimeSimulation(0)

        # camera  –  positioned to see the handover workspace
        p.resetDebugVisualizerCamera(2.0, 50, -20, [0.5, 0.1, 0.2])

        self.tb6       = None      # TB6 body id
        self.robotiq   = None      # Robotiq 3F body id
        self.rq_jidx   = {}        # Robotiq joint name → index
        self.jidx      = {}        # TB6   joint name → index
        self.flange_idx = None     # TB6 ee_Link  link index

        self.object_vis  = None    # held object visual
        self.hand_proxy  = None    # human hand visual
        self.rq_constraint = None  # fixed constraint  (TB6 flange → Robotiq)

        self.ft     = AnalyticalFTSensor()
        self.model  = BayesianContactModel()

        # state
        self.probing  = True
        self.contact  = 'none'
        self.t        = 0.0
        self.u_z      = 0.0
        self.base_joints = None    # nominal joint configuration (rad)
        self.log = deque(maxlen=10000)

        # pre-compute mounting quaternion
        self.mount_orn_offset = p.getQuaternionFromEuler(MOUNT_EULER)

    # ==================================================================
    def setup(self):
        # ---- ground (visual only, lower so it doesn't block view) ----
        p.loadURDF("plane.urdf", [0, 0, -0.2])

        # ---- load TB6  (fixed base) ----
        self.tb6 = p.loadURDF(URDF_TB6, [0, 0, 0], [0, 0, 0, 1],
                              useFixedBase=True)
        for i in range(p.getNumJoints(self.tb6)):
            self.jidx[p.getJointInfo(self.tb6, i)[1].decode()] = i

        # ---- natural handover pose ----
        # Arm extended forward, elbow down, palm ~50 cm out at chest height.
        # These angles were verified visually in the demo.
        self.flange_idx = p.getNumJoints(self.tb6) - 1
        self.base_joints = np.array(
            [-0.354, -0.901, 1.886, -2.149, 0.354, 3.133])
        for i, a in enumerate(self.base_joints):
            p.resetJointState(self.tb6, i, a)
        for _ in range(10):
            p.stepSimulation()

        st = p.getLinkState(self.tb6, self.flange_idx)
        flange_pos, flange_orn = np.array(st[0]), np.array(st[1])
        print(f"TB6 base joints: {np.round(self.base_joints,3).tolist()}")
        print(f"TB6 TCP: ({flange_pos[0]:.3f}, "
              f"{flange_pos[1]:.3f}, {flange_pos[2]:.3f})")

        # ---- Robotiq 3F (near-zero mass) ----
        rq_init_pos, rq_init_orn = _multiply_pose(
            flange_pos, flange_orn,
            MOUNT_TRANSLATION, self.mount_orn_offset)
        self.robotiq, self.rq_jidx = load_robotiq_kinematic(rq_init_pos,
                                                            rq_init_orn)
        open_gripper_urdf(self.robotiq, self.rq_jidx)
        self.rq_constraint = None

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
        for i in range(5):
            tid = p.addUserDebugText("", [0, 0, 1.3 - i * 0.08],
                                     textColorRGB=[1, 1, 1], textSize=1.8)
            self.hud.append(tid)
        self._redraw()

        print("Ready.  p=probe  g=grasp  t=touch  n=none  o/c=gripper  "
              "r=reset  ESC=quit")

    # ==================================================================
    def _sync_rq_to_flange(self):
        """Place Robotiq at correct world pose relative to TB6 flange."""
        fs = p.getLinkState(self.tb6, self.flange_idx)
        rpos, rorn = _multiply_pose(fs[0], fs[1],
                                    MOUNT_TRANSLATION, self.mount_orn_offset)
        p.resetBasePositionAndOrientation(self.robotiq, rpos, rorn)

    # ==================================================================
    def _get_palm_pose(self):
        """Return (pos, orn) of Robotiq palm link in world frame."""
        pos, orn = p.getBasePositionAndOrientation(self.robotiq)
        return np.array(pos), np.array(orn)

    # ==================================================================
    def _sync_attachments(self):
        """Update object and hand proxy positions to track the palm frame."""
        palm_pos, palm_orn = self._get_palm_pose()

        obj_pos, obj_orn = _multiply_pose(palm_pos, palm_orn,
                                          OBJECT_OFFSET, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.object_vis, obj_pos, obj_orn)

        # Hand position depends on contact type
        if self.contact == 'none':
            h_off = [HAND_OFFSET[0], HAND_OFFSET[1] - 0.15, HAND_OFFSET[2]]
        elif self.contact == 'touch':
            h_off = [HAND_OFFSET[0], HAND_OFFSET[1] + 0.01, HAND_OFFSET[2] - 0.06]
        else:  # grasp
            h_off = [HAND_OFFSET[0], HAND_OFFSET[1] + 0.03, HAND_OFFSET[2]]

        hand_pos, hand_orn = _multiply_pose(palm_pos, palm_orn,
                                            h_off, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.hand_proxy, hand_pos, hand_orn)

    # ==================================================================
    def _redraw(self):
        cn = {'none': 'NONE', 'touch': 'TOUCH', 'grasp': 'FIRM GRASP'}
        firm = self.model.check_firm_grasp()
        lines = [
            f"Contact: {cn[self.contact]}    "
            f"Probing: {'ON' if self.probing else 'OFF'}",
            f">>> Firm Grasp: {'YES!' if firm else 'no'} <<<",
            f"Model: H(w)={self.model.entropy():.1f}  "
            f"w=[{self.model.m[0]:.0f}, {self.model.m[1]:.0f}]",
            f"---",
            f"p=toggle probe  g=grasp  t=touch  n=none  "
            f"o/c=grip  r=reset  ESC=quit",
        ]
        colors = [
            [0.3, 1, 0.3] if self.contact == 'grasp' else
            [1, 0.5, 0.3] if self.contact == 'touch' else [0.7, 0.7, 0.7],
            [0.2, 1, 0.2] if firm else [1, 0.4, 0.4],
            [1, 1, 0.7],
            [0.5, 0.5, 0.5],
            [0.5, 0.7, 1],
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
        # 1.  Compute joint targets (probing or hold)
        if self.probing:
            phase = np.sin(2 * np.pi * PROBE_FREQ * self.t)
            targets = self.base_joints.copy()
            targets[1] += PROBE_AMP * phase
            targets[2] += PROBE_AMP * 0.6 * phase
            targets[4] -= PROBE_AMP * 0.3 * phase
            dphase_dt = (2 * np.pi * PROBE_FREQ *
                         np.cos(2 * np.pi * PROBE_FREQ * self.t))
            self.u_z = np.clip(PROBE_DELTA_Z * dphase_dt, -V_MAX, V_MAX)
        else:
            targets = self.base_joints
            self.u_z = 0.0

        # 2.  Set TB6 joints + step + sync Robotiq (240 Hz, proven pattern)
        for i in range(6):
            p.resetJointState(self.tb6, i, float(targets[i]))
        p.stepSimulation()
        self._sync_rq_to_flange()
        self._sync_attachments()

        # 3.  FT sensor + Bayesian model
        ft = self.ft.read(self.u_z)
        self.model.update(self.u_z, ft[2])
        self.t += 1.0 / SIM_FREQ

        r = {'t': self.t, 'u_z': self.u_z, 'f_z': ft[2],
             'contact': self.contact, 'probing': self.probing,
             'firm': self.model.check_firm_grasp(),
             'entropy': self.model.entropy(),
             'w_up': float(self.model.m[0]),
             'w_down': float(self.model.m[1])}
        self.log.append(r)
        return r

    # ==================================================================
    def run(self):
        print("\nProbing is ON by default.  Press keys to change contact "
              "type.\n")
        report = SIM_FREQ
        i = 0
        try:
            while True:
                # ----- keyboard -----
                try:
                    keys = p.getKeyboardEvents()
                except p.error:
                    break

                for k, v in keys.items():
                    if v & p.KEY_WAS_TRIGGERED:
                        if k == ord('p'):
                            self.probing = not self.probing
                            print(f"  PROBE: {'ON' if self.probing else 'OFF'}")
                        elif k == ord('g'):
                            self.contact = 'grasp'
                            self.ft.set_contact('grasp')
                            print("  CONTACT: FIRM GRASP")
                        elif k == ord('t'):
                            self.contact = 'touch'
                            self.ft.set_contact('touch')
                            print("  CONTACT: INCIDENTAL TOUCH")
                        elif k == ord('n'):
                            self.contact = 'none'
                            self.ft.set_contact('none')
                            print("  CONTACT: NONE")
                        elif k == ord('o'):
                            open_gripper_urdf(self.robotiq, self.rq_jidx)
                            print("  GRIPPER: OPEN")
                        elif k == ord('c'):
                            close_gripper_urdf(self.robotiq, self.rq_jidx)
                            print("  GRIPPER: CLOSE")
                        elif k == ord('r'):
                            self.model.reset()
                            print("  MODEL RESET")
                        elif k == 27:
                            print("Quit.")
                            self._save_log()
                            p.disconnect()
                            return

                # ----- simulation step -----
                r = self.step()
                if r is None:
                    break

                if i % (SIM_FREQ // 2) == 0:
                    self._redraw()

                if i % report == 0:
                    firm_str = '*** FIRM! ***' if r['firm'] else '-'
                    print(f"t={r['t']:5.1f}s | u={r['u_z']:+5.2f} "
                          f"fz={r['f_z']:+6.2f}N | firm={firm_str:14s} "
                          f"w=[{r['w_up']:5.0f},{r['w_down']:5.0f}] "
                          f"ent={r['entropy']:.1f}")

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
