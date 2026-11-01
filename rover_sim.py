"""
rover_sim.py — MuJoCo 上の差動二輪ロボットと、目印へ向かう簡単な制御・録画

構成
  - 床、色つきの目印（赤い柱・青い柱・緑の箱・黄色い箱）
  - 差動二輪ロボット：左右の車輪を速度制御、後ろにキャスター（球）
  - 行動計画 = [{"action":"goto","target":"red_pillar"}, {"action":"turn","deg":90}, {"action":"wait","sec":1}, ...]
    を順に実行し、追従カメラの映像を mp4 に保存する
"""
import math
import numpy as np
import mujoco
import imageio

# 目印の名前（LLMにもこの名前を渡す）と、位置・色
LANDMARKS = {
    "red_pillar":   {"pos": (3.0, 0.0),   "rgba": "0.9 0.2 0.2 1", "shape": "cylinder", "radius": 0.12, "ja": "赤い柱"},
    "blue_pillar":  {"pos": (0.0, 3.0),   "rgba": "0.2 0.3 0.9 1", "shape": "cylinder", "radius": 0.12, "ja": "青い柱"},
    "green_box":    {"pos": (-3.0, 0.0),  "rgba": "0.2 0.7 0.3 1", "shape": "box",      "radius": 0.36, "ja": "緑の箱"},
    "yellow_box":   {"pos": (0.0, -3.0),  "rgba": "0.9 0.8 0.2 1", "shape": "box",      "radius": 0.36, "ja": "黄色い箱"},
}

WHEEL_R = 0.08     # 車輪半径 [m]
WHEEL_L = 0.30     # 左右車輪の間隔 [m]
V_MAX = 0.6        # 最大前進速度 [m/s]
W_MAX = 1.5        # 最大旋回速度 [rad/s]


def build_xml():
    marks = []
    for name, m in LANDMARKS.items():
        x, y = m["pos"]
        if m["shape"] == "cylinder":
            geom = f'<geom name="{name}" type="cylinder" size="0.12 0.5" pos="{x} {y} 0.5" rgba="{m["rgba"]}"/>'
        else:
            geom = f'<geom name="{name}" type="box" size="0.25 0.25 0.25" pos="{x} {y} 0.25" rgba="{m["rgba"]}"/>'
        marks.append(geom)
    marks = "\n      ".join(marks)
    return f"""
<mujoco model="rover">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="480"/><headlight diffuse="0.3 0.3 0.3" ambient="0.3 0.3 0.3" specular="0 0 0"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.55 0.58 0.62" rgb2="0.42 0.45 0.5" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0.05"/>
  </asset>
  <default>
    <geom condim="3" friction="1.0 0.005 0.0001"/>
  </default>
  <worldbody>
    <light pos="0 0 6" dir="0.2 0.2 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="8 8 0.1" material="grid"/>
    {marks}
    <body name="chassis" pos="0 0 0.12">
      <freejoint name="root"/>
      <geom name="body" type="box" size="0.18 0.12 0.04" rgba="0.25 0.3 0.4 1" mass="3.0"/>
      <geom name="nose" type="box" size="0.03 0.06 0.02" pos="0.19 0 0.03" rgba="1 0.5 0 1" mass="0.05"/>
      <site name="center" pos="0 0 0" size="0.01"/>
      <camera name="chase" mode="trackcom" pos="-2.2 0 1.4" xyaxes="0 -1 0 0.5 0 1"/>
      <body name="wheel_l" pos="0 {WHEEL_L/2} -0.02">
        <joint name="jl" type="hinge" axis="0 1 0" armature="0.01" damping="0.05"/>
        <geom type="cylinder" size="{WHEEL_R} 0.02" euler="90 0 0" rgba="0.1 0.1 0.1 1" mass="0.3"/>
      </body>
      <body name="wheel_r" pos="0 {-WHEEL_L/2} -0.02">
        <joint name="jr" type="hinge" axis="0 1 0" armature="0.01" damping="0.05"/>
        <geom type="cylinder" size="{WHEEL_R} 0.02" euler="90 0 0" rgba="0.1 0.1 0.1 1" mass="0.3"/>
      </body>
      <geom name="caster" type="sphere" size="0.04" pos="-0.15 0 -0.06" rgba="0.5 0.5 0.5 1" mass="0.1" friction="0.01 0.001 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="ml" joint="jl" kv="1.0" ctrlrange="-20 20"/>
    <velocity name="mr" joint="jr" kv="1.0" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""


_MODEL, _RENDERER = None, None   # 描画器は1プロセスに1つだけ作る（複数作ると画面が乱れる環境があるため）


class Rover:
    def __init__(self, width=640, height=480, fps=25):
        global _MODEL, _RENDERER
        if _MODEL is None:
            _MODEL = mujoco.MjModel.from_xml_string(build_xml())
            _RENDERER = mujoco.Renderer(_MODEL, height, width)
        self.model = _MODEL
        self.renderer = _RENDERER
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "chase")
        self.fps = fps
        self.steps_per_frame = int(round(1.0 / (fps * self.model.opt.timestep)))
        self.frames = []
        self.trace = []           # (t, x, y, yaw) の記録。レポート用
        self.t = 0.0
        mujoco.mj_forward(self.model, self.data)

    # --- 状態 ---
    def pose(self):
        x, y = self.data.qpos[0], self.data.qpos[1]
        w, qx, qy, qz = self.data.qpos[3:7]
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return x, y, yaw

    # --- 低レベル：速度指令 → 車輪回転速度 ---
    def drive(self, v, w):
        v = float(np.clip(v, -V_MAX, V_MAX))
        w = float(np.clip(w, -W_MAX, W_MAX))
        self.data.ctrl[0] = (v - w * WHEEL_L / 2) / WHEEL_R   # 左
        self.data.ctrl[1] = (v + w * WHEEL_L / 2) / WHEEL_R   # 右

    def step(self, n=1, record=True):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self.t += self.model.opt.timestep
            k = int(round(self.t / self.model.opt.timestep))
            if record and k % self.steps_per_frame == 0:
                self.renderer.update_scene(self.data, camera=self.cam)
                self.frames.append(self.renderer.render().copy())
                self.trace.append((self.t, *self.pose()))

    # --- 中レベル：行動 ---
    def run_for(self, v, w, sec):
        self.drive(v, w)
        self.step(int(sec / self.model.opt.timestep))

    def goto(self, target, margin=0.35, timeout=25.0):
        """目印 target へ向かう。向きを合わせながら進む比例制御。戻り値: 到達したか"""
        tx, ty = LANDMARKS[target]["pos"]
        stop_dist = LANDMARKS[target]["radius"] + margin + 0.2   # 目印の半径 + 余裕 + 車体の長さ
        t0 = self.t
        while self.t - t0 < timeout:
            x, y, yaw = self.pose()
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if dist < stop_dist:
                self.drive(0, 0); self.step(int(0.3 / self.model.opt.timestep))
                return True
            err = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi   # -pi..pi
            w = 2.0 * err
            v = V_MAX * max(0.0, math.cos(err)) * min(1.0, dist / 0.8)              # 向きがずれていたら減速
            self.drive(v, w)
            self.step(10)
        self.drive(0, 0)
        return False

    def turn(self, deg):
        """その場で deg 度回る（正が左回り）"""
        x, y, yaw0 = self.pose()
        goal = yaw0 + math.radians(deg)
        t0 = self.t
        while self.t - t0 < 10.0:
            _, _, yaw = self.pose()
            err = (goal - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < math.radians(3):
                break
            self.drive(0, np.clip(2.5 * err, -W_MAX, W_MAX) if abs(err) > 0.2 else np.sign(err) * 0.4)
            self.step(10)
        self.drive(0, 0); self.step(int(0.3 / self.model.opt.timestep))

    def wait(self, sec):
        self.drive(0, 0); self.step(int(sec / self.model.opt.timestep))

    # --- 高レベル：計画の実行 ---
    def execute(self, plan):
        log = []
        for seg in plan:
            a = seg["action"]
            x0, y0, _ = self.pose(); t0 = self.t
            if a == "goto":
                ok = self.goto(seg["target"]); extra = {"reached": ok}
            elif a == "turn":
                self.turn(seg["deg"]); extra = {}
            elif a == "wait":
                self.wait(seg["sec"]); extra = {}
            elif a == "move":   # 前進 sec 秒
                self.run_for(seg.get("v", 0.4), 0.0, seg["sec"]); extra = {}
            else:
                extra = {"error": f"unknown action {a}"}
            x1, y1, yaw1 = self.pose()
            log.append({**seg, **extra, "t_start": round(t0, 2), "t_end": round(self.t, 2),
                        "pos_end": [round(x1, 2), round(y1, 2)], "yaw_end_deg": round(math.degrees(yaw1), 1)})
        return log

    def save_video(self, path):
        imageio.mimwrite(path, self.frames, fps=self.fps, codec="libx264", quality=8, macro_block_size=None)
        return path


if __name__ == "__main__":
    import sys, json, os
    out = sys.argv[1] if len(sys.argv) > 1 else "rover_test.mp4"
    r = Rover()
    plan = [{"action": "goto", "target": "red_pillar"}, {"action": "turn", "deg": 90},
            {"action": "goto", "target": "blue_pillar"}, {"action": "wait", "sec": 1.0},
            {"action": "goto", "target": "green_box"}]
    log = r.execute(plan)
    print(json.dumps(log, ensure_ascii=False, indent=1))
    r.save_video(out)
    print("video:", out, len(r.frames), "frames", os.path.getsize(out) // 1024, "KB")
