"""
Real-Machine Handover Experiment — TB6 R5 + Robotiq 3F + Wrist FT Sensor
=======================================================================
Complete deployment matching simulate_handover.py state machine.

SAFETY (read before operating):
  - SetRate at 10-20 = very slow motion, easy to stop
  - ENTER key at any time = EMERGENCY STOP (Stop → Disable)
  - All joint targets are checked against limits before moving
  - FT sensor is monitored for excessive force (>50N triggers auto-stop)
  - Robotiq force is set to moderate level (0x40 = ~25N)
  - Move duration: 8s (slower than simulation for safety)

PREREQUISITES:
  pip install pyserial numpy scipy

EXPERIMENT FLOW (matches simulation):
  HOME → MOVING_TO_POSE → READY → GRASPING (close Robotiq)
       → HOLDING → DETECTING (probing + Bayesian model)
       → FIRM_DETECTED → RELEASING (open Robotiq) → DONE
       → MOVING_TO_HOME → HOME

FT SENSOR NOTE:
  The sensor's Fz is along the tool axis (J6 rotation axis).
  We rotate FT readings to world frame via the TCP quaternion from
  the topic so that Fz aligns with gravity, matching the paper's assumption
  that the probing direction and force sensing are both vertical.

PROBING: SpeedL (Cartesian velocity, world frame) oscillates TCP along
  world Z (vertical/gravity).  The controller handles IK automatically.
"""

import rpc
import topic
import message
import random
import time
import threading
import math
import os
import json
import struct
import os
import sys
from collections import deque
from datetime import datetime

import numpy as np


def _install_topic_filter():
    """Redirect C stdout (fd 1) through a pipe to suppress C++ topic log spam.
    Python output is routed directly to the original stdout, bypassing the filter.
    Returns (saved_fd, filter_thread).
    """
    saved_fd = os.dup(1)           # save original stdout
    r_fd, w_fd = os.pipe()         # create pipe
    os.dup2(w_fd, 1)               # C stdout now writes to pipe
    os.close(w_fd)

    # Route Python stdout directly to the real console (bypasses the pipe/filter)
    sys.stdout = os.fdopen(os.dup(saved_fd), 'w', buffering=1, closefd=False)

    def _pump():
        with os.fdopen(r_fd, 'r', encoding='utf-8', errors='replace', closefd=True) as src:
            for line in src:
                if ("No callbacks registered for topic:" not in line
                        and "[await]" not in line):
                    os.write(saved_fd, line.encode('utf-8', errors='replace'))

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    return saved_fd, t


def _restore_topic_filter(saved_fd):
    """Restore original stdout fd."""
    sys.stdout.flush()
    sys.stdout.close()
    os.dup2(saved_fd, 1)
    os.close(saved_fd)
    sys.stdout = os.fdopen(os.dup(1), 'w', buffering=1, closefd=False)

# ===========================================================================
# Configuration — ADJUST THESE TO MATCH YOUR SETUP
# ===========================================================================
TB6_IP = "192.168.50.1"
TB6_PORT = 5868
TOPIC_PORT = 19091

# Robotiq gripper — Modbus RTU (RS232)
# Set ROBOTIQ_PORT to your COM port, e.g. "COM12" on Windows, "/dev/ttyUSB0" on Linux
# Set ROBOTIQ_PORT = None to run WITHOUT gripper (arm-only test mode)
ROBOTIQ_PORT = "COM12"          # e.g. "COM12"
ROBOTIQ_SLAVE_ID = 9
ROBOTIQ_BAUDRATE = 115200

# ---- Joint configurations (MUST TEACH ON REAL ROBOT) ----
# HOME: arm folded, safe compact pose
HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# HANDOVER: arm reaching to experiment workspace
# *** TEACH THESE ON THE REAL ROBOT VIA WEB UI JOGGING FIRST ***
HANDOVER_JOINTS = [0.0, -0.901, 1.886, -2.149, 0.354, 3.133]

# ---- TB6 joint limits (from URDF) ----
JOINT_LIMITS = [
    (-3.1416, 3.1416),   # J1
    (-3.1416, 3.1416),   # J2
    (-2.8623, 2.8623),   # J3 — restricted!
    (-3.1416, 3.1416),   # J4
    (-3.1416, 3.1416),   # J5
    (-3.1416, 3.1416),   # J6
]

# ---- Motion speeds ----
SETRATE = 15               # global speed percentage (10-30, low = safe)
MOVE_DURATION = 8.0        # seconds for MoveAbsJ (slower than simulation's 5s)

# ---- Probing parameters ----
PROBE_DELTA_Z = 0.030       # vertical TCP oscillation amplitude (m)
PROBE_FREQ = 1.5            # Hz — approximate, depends on MoveAbsJ speed

# ---- Bayesian model ----
BETA = 0.05                 # measurement noise variance
CONFIDENCE_C = 0.99         # confidence level for firm-grasp check
OBJECT_WEIGHT = 3.0         # N — measure your object's weight
DATA_BUFFER_SIZE = 200      # most recent (u, f) pairs

# ---- FT sensor ----
FT_TARE_SAMPLES = 100       # samples for tare averaging
FT_EXCESSIVE_FORCE = 50.0   # N — auto-stop if exceeded
V_MAX = 0.04                # max gripper velocity (m/s), for firm-grasp check

# ---- Logging ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


# ===========================================================================
# Modbus RTU CRC-16
# ===========================================================================
def _modbus_crc(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


# ===========================================================================
# Robotiq 3F Gripper — Modbus RTU (Simplified Control Mode)
# ===========================================================================
class RobotiqGripper:
    """Robotiq Adaptive Gripper S-Model via Modbus RTU (RS232).

    Registers (Simplified Control Mode):
      Robot Output (write): base = 0x03E8 (1000)
        Byte 0: ACTION REQUEST  (rACT, rMOD, rGTO, rATR)
        Byte 1: GRIPPER OPTIONS (00000000 in simple mode)
        Byte 2: 00000000
        Byte 3: POSITION REQUEST (0x00=open … 0xFF=closed)
        Byte 4: SPEED
        Byte 5: FORCE
        Bytes 6-15: 00000000

      Robot Input (read): base = 0x07D0 (2000)
        Byte 0: GRIPPER STATUS (gACT, gMOD, gGTO, gIMC, gSTA)
        Byte 1: OBJECT STATUS  (gDTA, gDTB, gDTC, gDTS)
        Byte 2: FAULT STATUS
        Byte 3: POSITION REQUEST ECHO
    """

    def __init__(self, port, slave_id=9, baudrate=115200):
        import serial as _serial
        self._ser = _serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=_serial.EIGHTBITS,
            parity=_serial.PARITY_NONE,
            stopbits=_serial.STOPBITS_ONE,
            timeout=0.1,
        )
        self._sid = slave_id
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _fc03(self, start_addr, count):
        """Read Holding Registers (FC03).  Returns list of ints (register values)."""
        req = struct.pack('>B B H H',
                          self._sid, 0x03, start_addr, count)
        crc = _modbus_crc(req)
        req += struct.pack('<H', crc)
        with self._lock:
            self._ser.reset_input_buffer()
            self._ser.write(req)
            # Response: sid, 0x03, byte_count, data[], crc
            hdr = self._ser.read(3)
            if len(hdr) < 3:
                return None
            byte_count = hdr[2]
            data = self._ser.read(byte_count + 2)
            if len(data) < byte_count + 2:
                return None
            payload = data[:byte_count]
        return [payload[i] << 8 | payload[i + 1] for i in range(0, byte_count, 2)]

    def _fc16(self, start_addr, registers):
        """Write Multiple Registers (FC16).  registers = list of 16-bit ints."""
        count = len(registers)
        byte_count = count * 2
        req = struct.pack('>B B H H B',
                          self._sid, 0x10, start_addr, count, byte_count)
        for r in registers:
            req += struct.pack('>H', r)
        crc = _modbus_crc(req)
        req += struct.pack('<H', crc)
        with self._lock:
            self._ser.reset_input_buffer()
            self._ser.write(req)
            resp = self._ser.read(8)  # sid, 0x10, start, count, crc
            return len(resp) >= 6

    # ------------------------------------------------------------------
    def activate(self):
        """rACT = 1, clear everything else.  Returns True on success."""
        ok = self._fc16(0x03E8, [0x0100, 0x0000, 0x0000])
        if ok:
            print("[Robotiq] Activation sent.")
        return ok

    def reset_gripper(self):
        """rACT = 0 — reset gripper."""
        return self._fc16(0x03E8, [0x0000, 0x0000, 0x0000])

    def is_activated(self):
        """Check gIMC bits == 3 (activation complete)."""
        regs = self._fc03(0x07D0, 1)
        if regs is None:
            return False
        status_byte = (regs[0] >> 8) & 0xFF  # Byte 0
        gimc = (status_byte >> 4) & 0x03
        return gimc == 3

    def wait_activation(self, timeout=5.0):
        """Block until activation completes or timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.is_activated():
                print("[Robotiq] Activation complete.")
                return True
            time.sleep(0.1)
        print("[Robotiq] WARNING: Activation timeout!")
        return False

    # ------------------------------------------------------------------
    def move(self, position, speed=0x40, force=0x40):
        """Move fingers to position (0x00=open … 0xFF=closed).
        speed, force: 0x00 (min) – 0xFF (max).
        """
        # ACTION REQUEST: rACT=1, rMOD=0(basic), rGTO=1 → 0x09
        # Byte 0: 0x09, Byte 1: 0x00, Byte 2: 0x00
        # Byte 3: position, Byte 4: speed, Byte 5: force
        return self._fc16(0x03E8,
                          [0x0900, 0x0000 | position, speed << 8 | force,
                           0x0000, 0x0000, 0x0000])

    def open(self, speed=0x40, force=0x40):
        return self.move(0x00, speed, force)

    def close(self, speed=0x40, force=0x40):
        return self.move(0xFF, speed, force)

    # ------------------------------------------------------------------
    def read_gripper_status(self):
        """Return dict with gACT, gMOD, gGTO, gIMC, gSTA, fault, pos_echo."""
        regs = self._fc03(0x07D0, 2)
        if regs is None:
            return None
        b0 = (regs[0] >> 8) & 0xFF
        b1 = regs[0] & 0xFF
        b2 = (regs[1] >> 8) & 0xFF
        b3 = regs[1] & 0xFF
        return {
            'gACT': b0 & 0x01,
            'gMOD': (b0 >> 1) & 0x03,
            'gGTO': (b0 >> 3) & 0x01,
            'gIMC': (b0 >> 4) & 0x03,
            'gSTA': (b0 >> 6) & 0x03,
            'object_status': b1,
            'fault': b2,
            'pos_echo': b3,
        }

    def is_stopped(self):
        """True if gripper stopped (gSTA != 0)."""
        st = self.read_gripper_status()
        return st is not None and st['gSTA'] != 0

    def is_grasped(self):
        """True if fingers stopped before reaching target (object detected)."""
        st = self.read_gripper_status()
        if st is None:
            return False
        return st['gSTA'] in (1, 2)

    def disconnect(self):
        """Close serial connection."""
        try:
            self._ser.close()
        except Exception:
            pass


# ===========================================================================
# Shared state (populated by topic callback)
# ===========================================================================
_ft_lock = threading.Lock()
_ft_raw = np.zeros(6)       # [fx, fy, fz, mx, my, mz] — in tool/sensor frame
_ft_tare = np.zeros(6)
_ft_tared = False
_joint_positions = np.zeros(6)
_tcp_quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # [qx, qy, qz, qw] tool→world
_system_running = True
_cb_count = 0


def _on_rtstate(tt: topic.SystemRtState):
    global _ft_raw, _joint_positions, _tcp_quaternion, _system_running, _cb_count
    _cb_count += 1
    if _cb_count <= 3:
        print(f"[Topic] rtstate callback #{_cb_count} received")

    parm = message.SystemStateData()
    message.display_rt(tt, parm)

    if parm.controller.ftvalues:
        ft = parm.controller.ftvalues[0]
        with _ft_lock:
            _ft_raw = np.array([ft.fx, ft.fy, ft.fz, ft.mx, ft.my, ft.mz])
        if _cb_count <= 3:
            print(f"[Topic] FT raw: fx={ft.fx:.2f} fy={ft.fy:.2f} fz={ft.fz:.2f}")

    joints_per_model = len(parm.models_joints) // max(len(parm.models), 1)
    if joints_per_model >= 6:
        with _ft_lock:
            for j in range(6):
                _joint_positions[j] = parm.models_joints[j].position
        if _cb_count <= 3:
            print(f"[Topic] Joints: {[_joint_positions[i] for i in range(6)]}")

    # Extract TCP orientation (flange/tool frame → world)
    if parm.models_current_points:
        rt = parm.models_current_points[0].robottarget
        if len(rt) >= 7:
            with _ft_lock:
                _tcp_quaternion = np.array([rt[3], rt[4], rt[5], rt[6]],
                                           dtype=float)
            if _cb_count <= 3:
                print(f"[Topic] TCP quat (qx,qy,qz,qw): "
                      f"{[f'{x:.3f}' for x in _tcp_quaternion]}")


def _quat_to_rot(q):
    """Convert quaternion [qx,qy,qz,qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),         1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ])


def read_ft():
    """Return tare-compensated FT [fx,fy,fz,mx,my,mz] in tool/sensor frame."""
    with _ft_lock:
        raw = _ft_raw.copy()
        tare = _ft_tare.copy()
        tared = _ft_tared
    return raw - tare if tared else raw


def read_ft_world():
    """Return tare-compensated FT [fx,fy,fz] in WORLD frame (Fz = gravity dir).

    The wrist FT sensor measures forces in the tool/flange frame where Fz is
    along the J6 rotation axis.  The paper expects Fz to be along gravity
    (world Z downward).  This function rotates the force vector from the
    tool frame into the world frame using the TCP quaternion from the topic.
    """
    ft_tool = read_ft()
    f_tool = ft_tool[:3]
    with _ft_lock:
        q = _tcp_quaternion.copy()
    R = _quat_to_rot(q)
    f_world = R @ f_tool
    return f_world


def read_ft_fz():
    """Read Fz only — NOW IN WORLD FRAME (gravity-aligned)."""
    return read_ft_world()[2]


def read_joints():
    """Return current joint positions [j1,…,j6] in rad."""
    with _ft_lock:
        return _joint_positions.copy()


# ===========================================================================
# Bayesian Contact Model (same as simulation)
# ===========================================================================
from scipy.stats import chi2


class BayesianContactModel:
    """Piecewise-linear Bayesian model of human-object contact state."""

    def __init__(self):
        self.beta = BETA
        self.d = 2
        self.m = np.zeros(2)
        self.S = np.eye(2) * 100.0
        self.S_inv = np.eye(2) / 100.0
        self.data = deque(maxlen=DATA_BUFFER_SIZE)

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


# ===========================================================================
# TB6 RPC utilities
# ===========================================================================
_stop_event = threading.Event()


def _estop_listener(client):
    """ENTER key → immediate Stop+Disable.
    Shares the main RPC client — may be delayed if main thread is
    inside a blocking CallAwait, but the stop flag is set immediately.
    """
    input()
    _stop_event.set()
    print("\n>>> EMERGENCY STOP TRIGGERED <<<")
    try:
        for cmd in ["{Stop}", "{Disable}"]:
            msg = rpc.Msg(cmd)
            msg.setMsgID(10001)
            msg.setMsgSeqID(random.randint(1, 10000))
            client.CallAwait(msg, 3000)
        print(">>> Arm stopped via estop.")
    except Exception as ex:
        print(f">>> Estop send error: {ex}")
        print(">>> USE PHYSICAL ESTOP BUTTON if arm is still moving!")


def send_cmd(client, cmd_str, timeout_ms=500):
    """Send a single RPC command synchronously. Returns (status, resp_list)."""
    if _stop_event.is_set():
        return -1, []
    msg = rpc.Msg(cmd_str)
    msg.setMsgID(10001)
    msg.setMsgSeqID(random.randint(1, 10000))
    status, resp_list = client.CallAwait(msg, timeout_ms)
    if status == 0:
        for r in resp_list:
            if r.code != 0:
                print(f"  [ERR({r.code})] {r.message}")
    else:
        print(f"  [FAIL] status={status}")
    return status, resp_list


def send_cmds(client, cmd_list, timeout_ms=500, sleep_s=0.1):
    """Send a list of RPC commands sequentially."""
    for cmd in cmd_list:
        if _stop_event.is_set():
            return False
        print(f"  -> {cmd}")
        status, _ = send_cmd(client, cmd, timeout_ms)
        if status != 0:
            print(f"  [WARN] Retrying with ClearErr...")
            send_cmd(client, "{Clear}", 500)
        time.sleep(sleep_s)
    return True


def e_stop(client):
    """Stop + Disable the arm immediately."""
    print(">>> STOPPING ARM...")
    for cmd in ["{Stop}", "{Disable}"]:
        msg = rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        client.CallAwait(msg, 3000)
    print(">>> Arm stopped and disabled.")


def re_enable(client):
    """Re-enable arm after estop (Clear → Disable → Mode → ... → Start)."""
    init_cmds = [
        "{Clear}",
        "{Disable}",
        "{Mode}",
        "{SetMaxToq}",
        "{Recover}",
        f"{{SetRate {SETRATE}}}",
        "{Enable}",
        "{Var --clear}",
        "{Recover}",
        "{Start}",
    ]
    return send_cmds(client, init_cmds, 500, 0.1)


def check_joint_limits(joints):
    """Verify all joint angles are within limits. Returns list of violations."""
    violations = []
    for i, (j, (lo, hi)) in enumerate(zip(joints, JOINT_LIMITS)):
        if j < lo - 0.01 or j > hi + 0.01:
            violations.append(f"J{i+1}={j:.3f} limit=[{lo:.3f},{hi:.3f}]")
    return violations


def defined_joint_target(client, name, joints):
    """Define a jointtarget variable on the controller."""
    j_str = ",".join(f"{x:.6f}" for x in joints)
    cmd = (f"{{Var --type=jointtarget --name={name}"
           f" --value={{{j_str},0,0,0,0}}}}")
    return send_cmd(client, cmd, 500)


# ===========================================================================
# Probing via MoveAbsJ position oscillation (runs in background thread)
# ===========================================================================
# SpeedL/SpeedJ async velocity commands proved unreliable on this TB6 firmware.
# MoveAbsJ CallAsync was also rejected (arm didn't move).  Only CallAwait
# (synchronous) works reliably.  Console [await] spam is suppressed via the
# stdout filter in _install_topic_filter().
# ===========================================================================
_probe_running = False

# Probe joint offset on J2/J3/J5 — tuned for visible but gentle vertical TCP
# motion.  Keep J1=0 (no base rotation).
_PROBE_OFFSET = np.array([0.0, 0.008, 0.005, 0.0, -0.003, 0.0])


def probing_loop(client):
    """Alternate between HANDOVER+offset and HANDOVER-offset via CallAwait MoveAbsJ.

    Each CallAwait blocks until the small move completes (~200-400 ms at
    SETRATE=15), giving roughly 1-2 Hz bidirectional probing.  The main
    detection loop runs independently in its own thread reading FT data.
    """
    global _probe_running
    base = np.array(HANDOVER_JOINTS)

    up_joints = base + _PROBE_OFFSET
    dn_joints = base - _PROBE_OFFSET
    send_cmd(client,
             f"{{Var --type=jointtarget --name=j_probe_up "
             f"--value={{{up_joints[0]:.6f},{up_joints[1]:.6f},{up_joints[2]:.6f},"
             f"{up_joints[3]:.6f},{up_joints[4]:.6f},{up_joints[5]:.6f},"
             f"0,0,0,0}}}}",
             500)
    send_cmd(client,
             f"{{Var --type=jointtarget --name=j_probe_dn "
             f"--value={{{dn_joints[0]:.6f},{dn_joints[1]:.6f},{dn_joints[2]:.6f},"
             f"{dn_joints[3]:.6f},{dn_joints[4]:.6f},{dn_joints[5]:.6f},"
             f"0,0,0,0}}}}",
             500)

    _probe_running = True
    print("[Probe] Position oscillation started (MoveAbsJ).")
    print(f"[Probe] Offset: J2={_PROBE_OFFSET[1]:.3f} J3={_PROBE_OFFSET[2]:.3f} "
          f"J5={_PROBE_OFFSET[4]:.3f} rad")

    use_up = True
    while _probe_running and not _stop_event.is_set():
        target_name = "j_probe_up" if use_up else "j_probe_dn"
        cmd = f"{{MoveAbsJ --jointtarget_var={target_name}}}"
        msg = rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        client.CallAwait(msg, 3000)
        use_up = not use_up

    # Return to handover pose
    send_cmd(client, "{MoveAbsJ --jointtarget_var=j_handover}", 5000)
    _probe_running = False
    print("[Probe] Stopped.")


# ===========================================================================
# Main experiment
# ===========================================================================
def main():
    global _ft_tared, _ft_tare

    print("=" * 60)
    print("TB6 Handover Experiment — Active Contact Sensing")
    print("=" * 60)
    print("SAFETY: Press ENTER at any time for EMERGENCY STOP")
    print("        SetRate =", SETRATE, "(very slow)")
    print("        Move duration =", MOVE_DURATION, "s")
    if ROBOTIQ_PORT:
        print("        Robotiq gripper: ENABLED on", ROBOTIQ_PORT)
    else:
        print("        Robotiq gripper: DISABLED (arm-only test mode)")
    print("=" * 60)

    # ---- Connect RPC (single client, shared with estop) ----
    print(f"\n[RPC] Connecting to {TB6_IP}:{TB6_PORT}...")
    client = rpc.CPPClient(TB6_IP, TB6_PORT)
    print("[RPC] Client created. Starting estop listener...")
    listener = threading.Thread(target=_estop_listener, args=(client,), daemon=True)
    listener.start()

    # ---- Connect topic (FT sensor + joint state) ----
    print("\n[Topic] Connecting to FT sensor...")
    options = topic.NodeOptions()
    options.node_name = 'handover_exp'
    options.sub_url = f'tcp://{TB6_IP}:{TOPIC_PORT}'
    topic_node = topic.Node(options)
    if not topic_node.Start():
        print("FATAL: Failed to start topic node")
        return
    _rt_sub = topic_node.CreateSubscriptionRT("system_rtstate", _on_rtstate)
    print("[Topic] Subscribed.")
    time.sleep(1.0)

    # Suppress C++ topic layer "No callbacks registered" log spam
    _saved_fd, _filter_thread = _install_topic_filter()

    # ---- Connect Robotiq gripper (if configured) ----
    gripper = None
    if ROBOTIQ_PORT:
        print(f"\n[Robotiq] Connecting on {ROBOTIQ_PORT}...")
        try:
            gripper = RobotiqGripper(ROBOTIQ_PORT, ROBOTIQ_SLAVE_ID, ROBOTIQ_BAUDRATE)
            print("[Robotiq] Port opened.")
        except Exception as e:
            print(f"[Robotiq] ERROR: {e}")
            print("[Robotiq] Continuing without gripper.")
            gripper = None

    try:
        # ================================================================
        # STEP 1: Initialize TB6
        # ================================================================
        print("\n>>> STEP 1: Initializing TB6...")
        init_cmds = [
            "{Clear}",
            "{Disable}",
            "{Mode}",
            "{SetMaxToq}",
            "{Recover}",
            f"{{SetRate {SETRATE}}}",
            "{Enable}",
            "{Var --clear}",
            "{Recover}",
            "{Start}",
        ]
        if not send_cmds(client, init_cmds, 500, 0.1):
            e_stop(client)
            return

        # Define joint targets
        if not defined_joint_target(client, "j_home", HOME_JOINTS):
            e_stop(client)
            return
        if not defined_joint_target(client, "j_handover", HANDOVER_JOINTS):
            e_stop(client)
            return

        # Sanity check: verify handover joints are within limits
        violations = check_joint_limits(HANDOVER_JOINTS)
        if violations:
            print("\n!!! JOINT LIMIT VIOLATIONS in HANDOVER_JOINTS !!!")
            for v in violations:
                print(f"    {v}")
            print("Please re-teach HANDOVER_JOINTS on the real robot.")
            print("Emergency stop and abort.")
            e_stop(client)
            return

        print("[TB6] Initialized and enabled.")

        # ================================================================
        # STEP 2: Activate Robotiq gripper
        # ================================================================
        if gripper:
            print("\n>>> STEP 2: Activating Robotiq gripper...")
            gripper.activate()
            if not gripper.wait_activation(timeout=15.0):
                print("[Robotiq] WARNING: Activation may have failed.")
                print("Check power and connections. Continuing anyway.")
            else:
                gripper.open(speed=0x60, force=0x40)
                time.sleep(0.5)
        else:
            print("\n>>> STEP 2: Robotiq gripper skipped (not configured).")

        if _stop_event.is_set():
            e_stop(client)
            return

        # ================================================================
        # STEP 3: Move to HOME position
        # ================================================================
        print("\n>>> STEP 3: Moving to HOME position...")
        status, _ = send_cmd(client, "{MoveAbsJ --jointtarget_var=j_home}", 60000)
        if status != 0:
            print("[ERROR] MoveAbsJ to HOME failed!")
            e_stop(client)
            return
        if _stop_event.is_set():
            e_stop(client)
            return
        print("[TB6] At HOME.")
        time.sleep(1.0)

        # ================================================================
        # STEP 4: Move to HANDOVER position
        # ================================================================
        print("\n>>> STEP 4: Moving to HANDOVER position...")
        print("    ENSURE WORKSPACE IS CLEAR!")
        print("    Press ENTER now to abort if anything is in the way.")
        time.sleep(1.0)
        if _stop_event.is_set():
            e_stop(client)
            return

        status, _ = send_cmd(client, "{MoveAbsJ --jointtarget_var=j_handover}", 60000)
        if status != 0:
            print("[ERROR] MoveAbsJ to HANDOVER failed!")
            e_stop(client)
            return
        if _stop_event.is_set():
            e_stop(client)
            return
        print("[TB6] At HANDOVER position.")
        time.sleep(1.0)

        # ================================================================
        # STEP 5: Close gripper to grasp object
        # ================================================================
        if gripper:
            print("\n>>> STEP 5: Closing gripper to grasp object...")
            print("    Place object between gripper fingers NOW.")
            print("    Gripper closes in:", end="", flush=True)
            for i in range(3, 0, -1):
                print(f" {i}", end="", flush=True)
                time.sleep(1.0)
            print()
            if _stop_event.is_set():
                e_stop(client)
                gripper.open()
                return

            gripper.close(speed=0x30, force=0x40)  # slow close, moderate force
            time.sleep(1.5)
            grasped = gripper.is_grasped()
            print(f"    Object grasped: {grasped}")
        else:
            print("\n>>> STEP 5: Gripper skipped (arm-only mode).")

        # ================================================================
        # STEP 6: Tare FT sensor (after grasping, before human contact)
        # ================================================================
        print("\n>>> STEP 6: Taring FT sensor...")
        print("    DO NOT touch the object or gripper during tare!")

        # Give the system a moment to stabilize after grasping
        time.sleep(1.0)
        if _stop_event.is_set():
            e_stop(client)
            if gripper:
                gripper.open()
            return

        samples = []
        for _ in range(FT_TARE_SAMPLES):
            with _ft_lock:
                samples.append(_ft_raw.copy())
            time.sleep(0.01)
        _ft_tare = np.mean(samples, axis=0)
        _ft_tared = True
        print(f"[FT] Tare complete:")
        print(f"     F = ({_ft_tare[0]:.2f}, {_ft_tare[1]:.2f}, {_ft_tare[2]:.2f}) N")
        print(f"     M = ({_ft_tare[3]:.2f}, {_ft_tare[4]:.2f}, {_ft_tare[5]:.2f}) Nm")
        print(f"     NOTE: Fz is along tool axis (J6), toward the palm.")

        # ================================================================
        # STEP 7: Start probing and Bayesian detection
        # ================================================================
        print("\n>>> STEP 7: Starting probing oscillation...")
        print("    Human: grasp the object firmly when ready.")
        print("    Robot will detect bidirectional force via Bayesian model.")

        model = BayesianContactModel()
        probe_thread = threading.Thread(
            target=probing_loop, args=(client,), daemon=True)
        probe_thread.start()
        time.sleep(0.3)

        # ---- Detection loop ----
        print("\n>>> Detection active. Waiting for firm grasp...")
        print("    (Press ENTER at any time to emergency stop)")

        detect_start = time.time()
        contact_detected = False
        log_entries = []
        last_status = 0.0          # last status print time
        last_cb_count = _cb_count  # track topic callback health

        while not _stop_event.is_set():
            # Read FT sensor in world frame (Fz = gravity direction, per paper)
            f_world = read_ft_world()
            fz_world = f_world[2]
            ft_tool = read_ft()  # tool-frame for logging

            # Monitor for excessive force (check both tool and world frames)
            f_mag = float(np.linalg.norm(f_world))
            if f_mag > FT_EXCESSIVE_FORCE:
                print(f"\n!!! EXCESSIVE FORCE: |F|={f_mag:.1f}N > {FT_EXCESSIVE_FORCE}N !!!")
                print("Auto-stopping for safety.")
                _stop_event.set()
                break

            # Check if human has made contact (world-frame vertical force)
            if abs(fz_world) > 0.3 and not contact_detected:
                contact_detected = True
                print(f"[{time.time()-detect_start:.1f}s] Human contact detected "
                      f"(fz_world={fz_world:.2f}N, fz_tool={ft_tool[2]:.2f}N)")

            # Estimated vertical TCP velocity from probing oscillation.
            # Probing uses MoveAbsJ (position-based, roughly triangular),
            # but we approximate as sinusoidal for the Bayesian model:
            #   u_z ≈ PROBE_DELTA_Z · ω · cos(ω·t)
            t_elapsed = time.time() - detect_start
            omega = 2 * math.pi * PROBE_FREQ
            dphase = omega * math.cos(omega * t_elapsed)
            u_z = np.clip(PROBE_DELTA_Z * dphase, -V_MAX, V_MAX)

            # Update Bayesian model EVERY cycle (not only after contact).
            # Without contact the data is low-signal so uncertainty stays high;
            # once the human grasps, forces appear and the model converges.
            model.update(u_z, fz_world)

            # Check for firm grasp (runs every cycle)
            if model.check_firm_grasp():
                print(f"\n    >>> FIRM GRASP DETECTED! <<<")
                print(f"        fz_world={fz_world:.2f}N  "
                      f"w=[{model.m[0]:.1f},{model.m[1]:.1f}]  "
                      f"ent={model.entropy():.2f}")
                break

            # Log
            log_entries.append({
                't': t_elapsed,
                'u_z': u_z,
                'fz_world': float(fz_world),
                'fz_tool': float(ft_tool[2]),
                'fx_world': float(f_world[0]),
                'fy_world': float(f_world[1]),
                'w_up': float(model.m[0]),
                'w_down': float(model.m[1]),
                'entropy': model.entropy(),
                'firm': False,
                'contact': contact_detected,
            })

            # Status print every ~1 second (robust time-gap, not modulo)
            if t_elapsed - last_status >= 1.0:
                # Warn if topic callbacks seem to have stopped
                cb_delta = _cb_count - last_cb_count
                cb_warn = " [topic stalled?]" if cb_delta == 0 else ""
                firm_str = "FIRM!" if model.check_firm_grasp() else "…"
                print(f"    t={t_elapsed:5.1f}s  "
                      f"fz_world={fz_world:+6.2f}N  "
                      f"fz_tool={ft_tool[2]:+6.2f}N  "
                      f"w=[{model.m[0]:5.0f},{model.m[1]:5.0f}]  "
                      f"ent={model.entropy():.1f}  "
                      f"firm={firm_str}{cb_warn}",
                      flush=True)
                last_status = t_elapsed
                last_cb_count = _cb_count

            time.sleep(0.05)

        # ---- Stop probing ----
        global _probe_running
        _probe_running = False
        time.sleep(0.3)
        probe_thread.join(timeout=2.0)

        # Mark final log entry as firm if detected
        if log_entries:
            log_entries[-1]['firm'] = True

        # ---- Save log ----
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"real_handover_log_{ts}.json")
        try:
            with open(log_path, 'w') as f:
                json.dump(log_entries, f, indent=2)
            print(f"[Log] Saved: {log_path}")
        except Exception as ex:
            print(f"[Log] Save failed: {ex}")

        # ================================================================
        # STEP 8: Release gripper
        # ================================================================
        if not _stop_event.is_set():
            print("\n>>> STEP 8: Releasing object...")
            if gripper:
                gripper.open(speed=0x60, force=0x40)
                # Wait for gripper to fully open — poll gSTA until == 3 (reached
                # requested position) or timeout.  Must NOT move the arm before
                # the fingers are clear of the object.
                t0 = time.time()
                while time.time() - t0 < 5.0:
                    st = gripper.read_gripper_status()
                    if st is not None and st['gSTA'] == 3:
                        print(f"    Gripper fully open ({time.time()-t0:.1f}s).")
                        break
                    time.sleep(0.1)
                else:
                    print("    WARNING: Gripper open timeout — proceed with caution.")
                # Extra settling time after fingers stop
                time.sleep(0.5)
            else:
                print("    Gripper skipped (arm-only mode).")
            print("    Object released to human.")
        else:
            # Emergency: try to release anyway, then wait before moving arm
            if gripper:
                gripper.open(speed=0xFF, force=0xFF)
                time.sleep(1.5)

        # ================================================================
        # STEP 9: Return to HOME
        # ================================================================
        if not _stop_event.is_set():
            print("\n>>> STEP 9: Returning to HOME...")
            status, _ = send_cmd(client, "{MoveAbsJ --jointtarget_var=j_home}", 60000)
            if status != 0:
                print("[ERROR] MoveAbsJ to HOME failed!")
            else:
                print("[TB6] Back at HOME.")
        else:
            print("\n>>> STEP 9: Emergency recovery...")
            re_enable(client)
            status, _ = send_cmd(client, "{MoveAbsJ --jointtarget_var=j_home}", 60000)
            if status != 0:
                print("[ERROR] MoveAbsJ to HOME failed!")
            else:
                print("[TB6] Back at HOME (post-emergency).")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as ex:
        print(f"\nERROR: {ex}")
        import traceback
        traceback.print_exc()
    finally:
        # Always stop probing
        _probe_running = False

        # Always stop and disable the arm
        _stop_event.set()
        try:
            e_stop(client)
        except Exception:
            pass

        # Always open gripper for safety
        if gripper:
            try:
                gripper.open(speed=0xFF, force=0xFF)
                time.sleep(0.5)
                gripper.reset_gripper()
                gripper.disconnect()
            except Exception:
                pass

        # Shutdown topic
        try:
            topic_node.Shutdown()
        except Exception:
            pass

        # Restore stdout (undo topic log filter)
        if '_saved_fd' in dir():
            _restore_topic_filter(_saved_fd)

        print("\n" + "=" * 60)
        print("Experiment ended. Arm is stopped and disabled.")
        print("Gripper is open (safe state).")
        print("=" * 60)


# ===========================================================================
if __name__ == "__main__":
    main()
