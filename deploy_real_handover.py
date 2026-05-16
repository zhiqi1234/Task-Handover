"""
Real-Machine Handover Test — TB6 R5 + wrist FT sensor
======================================================
Based on verified working example: Hello_MoveAbsJ_win_py/main.py

SAFETY:
  - SetRate without value = slowest possible motion
  - Press ENTER at any time = EMERGENCY STOP
  - Test WITHOUT gripper first, WITHOUT object
  - Verify all positions in Web UI jogging before running

Flow:
  1. Initialize (Clear → Disable → Mode → SetMaxToq → Recover → SetRate → Enable)
  2. Move to HOME
  3. Move to HANDOVER position
  4. Start probing oscillation (SpeedJ)
  5. Read FT sensor (topic subscription)
  6. Detect firm grasp → stop probing
  7. Return to HOME
"""

import rpc
import topic
import message
import random
import time
import threading
import math
import numpy as np

# ===========================================================================
# Configuration — ADJUST THESE
# ===========================================================================
TB6_IP = "192.168.50.1"
TB6_PORT = 5868
TOPIC_PORT = 19091

# HOME: arm folded, safe compact pose (all zeros = standard reference)
HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# HANDOVER: arm reaching to experiment workspace. *** TEACH ON REAL ROBOT ***
HANDOVER_JOINTS = [0.0, -0.901, 1.886, -2.149, 0.354, 3.133]

# Probing parameters (conservative — adjust after first safe test)
PROBE_AMP = 0.006          # oscillation amplitude (rad), half of simulation
PROBE_FREQ = 1.5           # Hz
PROBE_DT = 0.05            # SpeedJ update interval (s)

# FT sensor
FT_TARE_SAMPLES = 100
OBJECT_WEIGHT = 3.0         # N — your object's weight


# ===========================================================================
stop_event = threading.Event()

# Shared FT data (populated by topic callback)
ft_lock = threading.Lock()
ft_raw = np.zeros(6)
ft_tare = np.zeros(6)
ft_tared = False
joint_positions = np.zeros(6)


def wait_for_enter():
    input()
    stop_event.set()
    print("\n>>> EMERGENCY STOP TRIGGERED <<<")


def e_stop(client):
    print(">>> Sending Stop + Disable ...")
    for cmd in ["{Stop}", "{Disable}"]:
        msg = rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        client.CallAwait(msg, 3000)
    print(">>> Arm stopped and disabled.")


def send_cmd(client, cmd_str, timeout_ms=500):
    """Send a single RPC command, return (status, response_list)."""
    if stop_event.is_set():
        return -1, []
    msg = rpc.Msg(cmd_str)
    msg.setMsgID(10001)
    msg.setMsgSeqID(random.randint(1, 10000))
    status, resp_list = client.CallAwait(msg, timeout_ms)
    if status == 0:
        for r in resp_list:
            code = "OK" if r.code == 0 else f"ERR({r.code})"
            if r.code != 0:
                print(f"  [{code}] {r.message}")
    else:
        print(f"  [FAIL] status={status}")
    return status, resp_list


def send_cmds(client, cmd_list, timeout_ms=500, sleep_s=0.1):
    """Send a list of RPC commands sequentially."""
    for cmd in cmd_list:
        if stop_event.is_set():
            return
        print(f"  {cmd}")
        status, _ = send_cmd(client, cmd, timeout_ms)
        if status != 0:
            print(f"  [WARN] Retrying with ClearErr...")
            send_cmd(client, "{Clear}", 500)
        time.sleep(sleep_s)


# ===========================================================================
# Topic callback for FT sensor + joint state
# ===========================================================================
def on_rtstate(tt: topic.SystemRtState):
    global ft_raw, ft_tared, ft_tare, joint_positions
    parm = message.SystemStateData()
    message.display_rt(tt, parm)

    # FT sensor
    if parm.controller.ftvalues:
        ft = parm.controller.ftvalues[0]
        with ft_lock:
            ft_raw = np.array([ft.fx, ft.fy, ft.fz, ft.mx, ft.my, ft.mz])

    # Joint positions
    joints_per_model = len(parm.models_joints) // max(len(parm.models), 1)
    if joints_per_model >= 6:
        with ft_lock:
            for j in range(6):
                joint_positions[j] = parm.models_joints[j].position


def read_ft():
    """Return tare-compensated FT [fx,fy,fz,mx,my,mz]."""
    with ft_lock:
        raw = ft_raw.copy()
    if ft_tared:
        return raw - ft_tare
    return raw


def tare_ft():
    """Tare FT sensor. Call AFTER grasping, BEFORE human contact."""
    global ft_tared, ft_tare
    print("[FT] Taring sensor...")
    time.sleep(0.5)
    samples = []
    for _ in range(FT_TARE_SAMPLES):
        with ft_lock:
            samples.append(ft_raw.copy())
        time.sleep(0.01)
    ft_tare = np.mean(samples, axis=0)
    ft_tared = True
    print(f"[FT] Tare: F=({ft_tare[0]:.2f},{ft_tare[1]:.2f},{ft_tare[2]:.2f}) N")


# ===========================================================================
# Probing via SpeedJ (asynchronous, non-blocking)
# ===========================================================================
def probing_loop(client):
    """
    Run probing oscillation via SpeedJ commands.
    Sends sinusoidal velocity on J2, J3, J5 at ~20Hz.
    Can be interrupted by stop_event.
    """
    omega = 2 * math.pi * PROBE_FREQ
    t0 = time.time()
    print("[Probe] Starting oscillation...")

    while not stop_event.is_set():
        t = time.time() - t0
        phase = math.sin(omega * t)
        dphase = omega * math.cos(omega * t)

        # Velocity commands (rad/s) — matched to simulation ratios
        v2 = PROBE_AMP * dphase
        v3 = PROBE_AMP * 0.6 * dphase
        v5 = -PROBE_AMP * 0.3 * dphase

        cmd = (f"{{SpeedJ --vel={{{v2:.4f},{v3:.4f},0,0,{v5:.4f},0}}"
               f" --acc={{5,5,5,5,5,5}}"
               f" --dec={{5,5,5,5,5,5}}"
               f" --jerk={{10,10,10,10,10,10}}"
               f" --last_count={int(PROBE_DT * 1000)}}}")

        msg = rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        client.CallAsync(msg, 100)  # async — don't wait

        time.sleep(PROBE_DT)

    # Stop probing
    send_cmd(client, "{SpeedJ --stop}", 500)
    print("[Probe] Stopped.")


# ===========================================================================
# Main experiment
# ===========================================================================
def main():
    # ---- Start emergency-stop listener ----
    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    # ---- Start topic subscription ----
    print("[Topic] Starting...")
    options = topic.NodeOptions()
    options.node_name = 'handover_test'
    options.sub_url = f'tcp://{TB6_IP}:{TOPIC_PORT}'
    topic_node = topic.Node(options)
    if not topic_node.Start():
        print("FATAL: Failed to start topic node")
        return
    topic_node.CreateSubscriptionRT("system_rtstate", on_rtstate)
    print("[Topic] Subscribed to FT sensor + joint state.")
    time.sleep(0.5)

    # ---- Connect RPC ----
    print(f"[RPC] Connecting to {TB6_IP}:{TB6_PORT}...")
    client = rpc.CPPClient(TB6_IP, TB6_PORT)
    print("[RPC] Connected!")

    try:
        # ================================================================
        # STEP 1: Initialize
        # ================================================================
        print("\n>>> STEP 1: Initializing TB6...")
        init_cmds = [
            "{Clear}",
            "{Disable}",
            "{Mode}",
            "{SetMaxToq}",
            "{Recover}",
            "{SetRate}",         # no value = slowest (safest)
            "{Enable}",
            "{Var --clear}",
            "{Recover}",
        ]
        send_cmds(client, init_cmds, 500, 0.1)
        if stop_event.is_set():
            e_stop(client)
            return

        # Define joint targets
        j_home_str = ",".join(str(x) for x in HOME_JOINTS)
        j_handover_str = ",".join(str(x) for x in HANDOVER_JOINTS)
        jt_vars = [
            f"{{Var --type=jointtarget --name=j_home --value={{{j_home_str},0,0,0,0}}}}",
            f"{{Var --type=jointtarget --name=j_handover --value={{{j_handover_str},0,0,0,0}}}}",
        ]
        send_cmds(client, jt_vars, 500, 0.05)
        print("[TB6] Initialized and enabled.")

        # ================================================================
        # STEP 2: Move to HOME
        # ================================================================
        print("\n>>> STEP 2: Moving to HOME...")
        send_cmd(client, "{MoveAbsJ --jointtarget_var=j_home}", 30000)
        if stop_event.is_set():
            e_stop(client)
            return
        print("[TB6] At HOME.")
        time.sleep(1.0)

        # ================================================================
        # STEP 3: Move to HANDOVER position
        # ================================================================
        print("\n>>> STEP 3: Moving to HANDOVER position...")
        print("    (Ensure workspace is clear!)")
        send_cmd(client, "{MoveAbsJ --jointtarget_var=j_handover}", 30000)
        if stop_event.is_set():
            e_stop(client)
            return
        print("[TB6] At HANDOVER position.")
        time.sleep(1.0)

        # ================================================================
        # STEP 4: Tare FT sensor (simulates "after grasping object")
        # ================================================================
        print("\n>>> STEP 4: Taring FT sensor...")
        print("    (No gripper = taring at empty load)")
        tare_ft()
        if stop_event.is_set():
            e_stop(client)
            return

        # ================================================================
        # STEP 5: Start probing + detection
        # ================================================================
        print("\n>>> STEP 5: Starting probing oscillation...")
        print("    Human: reach for the flange and push/pull gently.")
        print("    Robot will detect bidirectional force.")
        probe_thread = threading.Thread(
            target=probing_loop, args=(client,), daemon=True)
        probe_thread.start()

        # ---- Detection loop ----
        print("\n>>> STEP 6: Detecting firm grasp...")
        window_up = []
        window_down = []
        firm_threshold = max(0.5 * OBJECT_WEIGHT, 1.0)  # N

        while not stop_event.is_set():
            ft = read_ft()
            fz = ft[2]

            if fz > 0.3:
                window_up.append(fz)
            elif fz < -0.3:
                window_down.append(abs(fz))

            window_up = window_up[-50:]
            window_down = window_down[-50:]

            up_ok = len(window_up) >= 10 and max(window_up) >= firm_threshold
            down_ok = len(window_down) >= 10 and max(window_down) >= firm_threshold

            if up_ok and down_ok:
                print(f"\n    >>> FIRM GRASP: up={max(window_up):.1f}N, "
                      f"down={max(window_down):.1f}N <<<")
                break

            # Status print every second
            if int(time.time() * 10) % 20 == 0:
                up_str = f"up max={max(window_up):.1f}N" if window_up else "up: -"
                down_str = f"down max={max(window_down):.1f}N" if window_down else "down: -"
                print(f"    fz={fz:+6.2f}N  {up_str:20s}  {down_str}")

            time.sleep(0.05)

        # ================================================================
        # STEP 7: Stop probing and return HOME
        # ================================================================
        stop_event.set()  # signal probing thread to stop
        time.sleep(0.5)
        probe_thread.join(timeout=2.0)

        print("\n>>> STEP 7: Returning to HOME...")
        # Re-enable if needed (emergency stop may have disabled)
        send_cmd(client, "{Clear}", 500)
        send_cmd(client, "{Disable}", 500)
        time.sleep(0.3)
        send_cmd(client, "{Mode}", 500)
        send_cmd(client, "{SetMaxToq}", 500)
        send_cmd(client, "{Recover}", 500)
        send_cmd(client, "{SetRate}", 500)
        send_cmd(client, "{Enable}", 500)
        send_cmd(client, "{Var --clear}", 500)
        send_cmd(client, "{Recover}", 500)
        send_cmd(client, f"{{Var --type=jointtarget --name=j_home --value={{{j_home_str},0,0,0,0}}}}", 500)
        time.sleep(0.3)

        send_cmd(client, "{MoveAbsJ --jointtarget_var=j_home}", 30000)
        print("[TB6] Back at HOME.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as ex:
        print(f"\nERROR: {ex}")
    finally:
        stop_event.set()
        try:
            e_stop(client)
        except Exception:
            pass
        try:
            topic_node.Shutdown()
        except Exception:
            pass
        print("\n=== Experiment ended. ===")


if __name__ == "__main__":
    print("=" * 60)
    print("TB6 Handover Test — Active Contact Sensing")
    print("Press ENTER at any time for EMERGENCY STOP")
    print("=" * 60)
    main()
