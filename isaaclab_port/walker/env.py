"""The official Box2D walker task as an Isaac Lab DirectRLEnv on Newton MJWarp.

Each world can run a different course. A world keeps only a short window of collision
boxes, rewritten every control step from per-course tables. MuJoCo computes world-body
geom poses once, so writing them directly is safe, and contact cost stays independent
of course length. Observation, reward and termination follow the official formulas.
The official motor, a velocity target with a torque limit, runs inside the solver as a
stiff implicit velocity actuator whose per-world force range is rewritten each step.
Known physics difference: MuJoCo's soft contacts and four 200 Hz substeps replace
Box2D's rigid solver at 50 Hz. The contact and joint-limit stiffness below were tuned
so that a policy trained in official Box2D keeps walking when replayed here.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import math
from pathlib import Path

import numpy as np
import torch
import warp as wp
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg
from isaaclab.utils.configclass import configclass

from . import spec, terrain

ASSET = Path(__file__).resolve().parents[1] / "assets" / "box2d_walker.usda"
LEG_JOINTS = ("left_hip", "left_knee", "right_hip", "right_knee")
ROOT_JOINTS = ("root_x", "root_z", "root_pitch")
# Authored sizes bound the broadphase; each world then shrinks its boxes to fit.
AUTHORED_HALF_SIZE = (1.0, 1.0, 1.2)
LANE_HALF_WIDTH = 1.0
HULL_SAMPLE_SPACING = 0.2
SUBSTEPS = 4
SOLVER_ITERATIONS = 30
SOLVER_LS_ITERATIONS = 12
CONTACT_SOLREF = (0.015, 1.5)
LIMIT_SOLREF = (0.02, 1.0)
# MuJoCo's solimp: dmin, dmax, width, midpoint, power.
STIFF_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
# MuJoCo contacts are soft; a foot within this of the ground counts as touching.
CONTACT_TOLERANCE = 0.02


class TerrainBank:
    """Compiled courses on the GPU, evicting the least recently used when full."""

    def __init__(self, capacity, device):
        sizes = terrain.table_sizes()
        self.capacity, self.device = capacity, device
        self.boxes = torch.zeros(capacity, 320, 6, device=device)
        self.segments = torch.zeros(capacity, 480, 4, device=device)
        self.box_start = torch.zeros(capacity, sizes["cells"], dtype=torch.int64, device=device)
        self.seg_start = torch.zeros(capacity, sizes["cells"], dtype=torch.int64, device=device)
        self.heights = torch.zeros(capacity, sizes["heights"], device=device)
        self.initial_force = torch.zeros(capacity, device=device)
        self.slots = OrderedDict()
        self.free = list(range(capacity - 1, -1, -1))

    def __contains__(self, key):
        return key in self.slots

    def add(self, config, seed, in_use=frozenset(), tables=None):
        """Slot for a course. Pass precompiled tables to keep this call short."""
        key = (config.key, int(seed))
        if key in self.slots:
            self.slots.move_to_end(key)
            return self.slots[key]
        if not self.free:
            victim = next((k for k, slot in self.slots.items() if slot not in in_use), None)
            if victim is None:
                raise RuntimeError("Terrain bank is full of courses in use; raise its capacity")
            self.free.append(self.slots.pop(victim))
        tables = tables or terrain.compile_course(terrain.generate_course(config, seed))
        slot = self.free.pop()
        self.boxes[slot] = torch.from_numpy(tables["boxes"]).to(self.device)
        self.segments[slot] = torch.from_numpy(tables["segments"]).to(self.device)
        self.box_start[slot] = torch.from_numpy(tables["box_start"].astype(np.int64)).to(self.device)
        self.seg_start[slot] = torch.from_numpy(tables["seg_start"].astype(np.int64)).to(self.device)
        self.heights[slot] = torch.from_numpy(tables["heights"]).to(self.device)
        self.initial_force[slot] = float(tables["initial_force"])
        self.slots[key] = slot
        return slot


def _hull_samples():
    points = []
    for (x0, y0), (x1, y1) in zip(spec.HULL_POLY, spec.HULL_POLY[1:] + spec.HULL_POLY[:1]):
        count = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / HULL_SAMPLE_SPACING))
        points += [(x0 + (x1 - x0) * i / count, y0 + (y1 - y0) * i / count) for i in range(count)]
    return points


@configclass
class WalkerEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 2000.0 / spec.FPS
    action_space = 4
    observation_space = 24
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1.0 / spec.FPS, render_interval=1, gravity=(0.0, 0.0, -spec.GRAVITY))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=0.0, replicate_physics=True, clone_in_fabric=False,
    )
    ui_window_class_type = None
    video_recorder = None
    # Implicit integration keeps this stable; Box2D's motor is effectively exact.
    motor_gain: float = 300.0
    terrain_capacity: int = 2048
    robot_cfg: ArticulationCfg | None = None


def make_cfg(*, num_envs, device="cuda:0", physics=None, terrain_capacity=2048):
    cfg = WalkerEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.sim.device = device
    cfg.terrain_capacity = terrain_capacity
    cfg.seed = 0
    cfg.episode_length_s = math.nextafter(2000 * cfg.decimation * cfg.sim.dt, 0.0)
    if physics is None:
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
        physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=192, nconmax=64, iterations=SOLVER_ITERATIONS,
                                                       ls_iterations=SOLVER_LS_ITERATIONS, integrator="implicitfast"),
                            num_substeps=SUBSTEPS, use_cuda_graph=device.startswith("cuda"))
    cfg.sim.physics = physics
    cfg.robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_0/Robot",
        spawn=UsdFileCfg(usd_path=str(ASSET)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={"root_x": spec.START_HULL[0], "root_z": spec.START_HULL[1], "root_pitch": -spec.START_HULL[2],
                       "left_hip": -spec.START_HIP[0], "left_knee": -spec.START_KNEE[0],
                       "right_hip": -spec.START_HIP[1], "right_knee": -spec.START_KNEE[1]},
            joint_vel={".*": 0.0},
        ),
        actuators={"legs": ImplicitActuatorCfg(
            joint_names_expr=list(LEG_JOINTS), stiffness=0.0, damping=cfg.motor_gain,
            effort_limit_sim=spec.MOTORS_TORQUE,
        )},
    )
    cfg.robot_cfg.prim_path = "/World/envs/env_.*/Robot"
    return cfg


class WalkerEnv(DirectRLEnv):
    """Set ``assigner`` to choose a course when a world resets: ids -> course slots."""

    cfg: WalkerEnvCfg

    def __init__(self, cfg: WalkerEnvCfg, render_mode=None, **kwargs):
        self.assigner = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.bank = TerrainBank(cfg.terrain_capacity, self.device)
        self.bank.add(terrain.EnvConfig(), 0)
        ids, names = self.robot.find_joints(list(ROOT_JOINTS), preserve_order=True)
        self._root_x, self._root_z, self._root_pitch = ids
        self._leg_ids, names = self.robot.find_joints(list(LEG_JOINTS), preserve_order=True)
        if names != list(LEG_JOINTS):
            raise RuntimeError(f"Unexpected joints: {names}")
        device = self.device
        self.actions = torch.zeros((self.num_envs, 4), device=device)
        self.world_terrain = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.prev_shaping = torch.zeros(self.num_envs, device=device)
        self.fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._numerical_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._hull_points = torch.tensor(_hull_samples(), device=device)
        self._speed_limits = torch.tensor((spec.SPEED_HIP, spec.SPEED_KNEE) * 2, device=device)
        self._slots = torch.arange(terrain.COLLISION_SLOTS, device=device)
        self._lidar_slots = torch.arange(terrain.LIDAR_SLOTS, device=device)
        angles = torch.tensor(spec.LIDAR_ANGLES, device=device)
        self._rays = torch.stack((torch.sin(angles), -torch.cos(angles)), dim=-1) * spec.LIDAR_RANGE
        self._bind_solver()
        self.reset_worlds(torch.arange(self.num_envs, device=device))

    def _bind_solver(self):
        from isaaclab_newton.physics import NewtonManager
        solver = NewtonManager._solver
        # Queued changes would otherwise apply on the first step and undo the writes below.
        with wp.ScopedDevice(self.device):
            for change in NewtonManager._model_changes:
                solver.notify_model_changed(change)
        NewtonManager._model_changes = set()
        model, data = solver.mjw_model, solver.mjw_data
        static = np.nonzero(model.geom_bodyid.numpy() == 0)[0]
        if len(static) != terrain.COLLISION_SLOTS:
            raise RuntimeError(f"Expected {terrain.COLLISION_SLOTS} world geoms, found {len(static)}")
        self._geoms = torch.as_tensor(static, device=self.device)
        self._geom_xpos = wp.to_torch(data.geom_xpos)
        self._geom_xmat = wp.to_torch(data.geom_xmat)
        self._geom_size = wp.to_torch(model.geom_size)
        if self._geom_size.shape[0] != self.num_envs:
            raise RuntimeError("geom_size is not per-world in this Newton version")
        wp.to_torch(model.geom_friction)[..., 0] = spec.LEG_FRICTION
        solimp = torch.tensor(STIFF_SOLIMP, device=self.device)
        wp.to_torch(model.geom_solref)[...] = torch.tensor(CONTACT_SOLREF, device=self.device)
        wp.to_torch(model.geom_solimp)[...] = solimp
        wp.to_torch(model.jnt_solref)[...] = torch.tensor(LIMIT_SOLREF, device=self.device)
        wp.to_torch(model.jnt_solimp)[...] = solimp
        self._forcerange = wp.to_torch(model.actuator_forcerange)
        # Actuators follow the joints' order, which is the action order, checked here.
        if model.nu != 4 or self._forcerange.shape[0] != self.num_envs:
            raise RuntimeError(f"Expected 4 per-world actuators, found {model.nu}, shape {self._forcerange.shape}")
        if model.actuator_trnid.numpy()[:, 0].tolist() != self._leg_ids:
            raise RuntimeError("Actuator order differs from the action order")
        model.actuator_forcelimited.assign([True] * 4)
        self._actuators = torch.arange(4, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        stage = self.sim.stage
        path = "/World/envs/env_0/Course"
        UsdGeom.Xform.Define(stage, path)
        material = UsdShade.Material.Define(stage, path + "/GroundMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(spec.LEG_FRICTION)
        physics_material.CreateDynamicFrictionAttr(spec.LEG_FRICTION)
        physics_material.CreateRestitutionAttr(0.0)
        for index in range(terrain.COLLISION_SLOTS):
            cube = UsdGeom.Cube.Define(stage, f"{path}/Slot_{index:02d}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, terrain.PARK_Z + index))
            cube.AddScaleOp().Set(Gf.Vec3f(*(2.0 * s for s in AUTHORED_HALF_SIZE)))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose="physics")
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

    def _cell(self, x):
        return torch.floor(x / terrain.CELL).long().clamp(0, self.bank.box_start.shape[1] - 1)

    def _write_terrain_window(self):
        q = self.robot.data.joint_pos.torch
        t = self.world_terrain
        start = self.bank.box_start[t, self._cell(q[:, self._root_x])]
        index = (start[:, None] + self._slots).clamp(max=self.bank.boxes.shape[1] - 1)
        boxes = self.bank.boxes[t[:, None], index]
        cx, cz, hx, hz, c, s = boxes.unbind(-1)
        geoms = self._geoms
        self._geom_xpos[:, geoms, 0] = cx
        self._geom_xpos[:, geoms, 2] = cz
        self._geom_size[:, geoms, 0] = hx
        self._geom_size[:, geoms, 1] = LANE_HALF_WIDTH
        self._geom_size[:, geoms, 2] = hz
        # Local x runs along the edge and local z along its normal, rotating about y.
        self._geom_xmat[:, geoms, 0, 0] = c
        self._geom_xmat[:, geoms, 0, 2] = -s
        self._geom_xmat[:, geoms, 1, 1] = 1.0
        self._geom_xmat[:, geoms, 2, 0] = s
        self._geom_xmat[:, geoms, 2, 2] = c
        self._geom_xmat[:, geoms, 0, 1] = 0.0
        self._geom_xmat[:, geoms, 1, 0] = 0.0
        self._geom_xmat[:, geoms, 1, 2] = 0.0
        self._geom_xmat[:, geoms, 2, 1] = 0.0

    def _pre_physics_step(self, actions):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self._write_terrain_window()

    def _apply_action(self):
        target = self._speed_limits * torch.sign(self.actions)
        self.robot.set_joint_velocity_target_index(target=-target, joint_ids=self._leg_ids)
        limit = spec.MOTORS_TORQUE * self.actions.abs()
        self._forcerange[:, self._actuators, 0] = -limit
        self._forcerange[:, self._actuators, 1] = limit

    def _height(self, x):
        cell = (x / terrain.HEIGHT_DX).long().clamp(0, self.bank.heights.shape[1] - 1)
        return self.bank.heights[self.world_terrain[:, None], cell]

    def _pose(self):
        q = self.robot.data.joint_pos.torch
        return (q[:, self._root_x], q[:, self._root_z], -q[:, self._root_pitch], -q[:, self._leg_ids])

    def _foot_contacts(self, x, z, angle, legs):
        contacts = []
        for hip, knee in ((legs[:, 0], legs[:, 1]), (legs[:, 2], legs[:, 3])):
            knee_x = x - spec.LEG_DOWN * torch.sin(angle) + spec.LEG_H * torch.sin(angle + hip)
            knee_z = z + spec.LEG_DOWN * torch.cos(angle) - spec.LEG_H * torch.cos(angle + hip)
            lower = angle + hip + knee
            c, s = torch.cos(lower), torch.sin(lower)
            half = spec.LOWER_W / 2
            # Bottom corners plus the mid-sides, in the lower leg's frame (y up from knee).
            local = ((-half, -spec.LEG_H), (half, -spec.LEG_H), (-half, -spec.LEG_H / 2), (half, -spec.LEG_H / 2))
            touching = torch.zeros_like(x, dtype=torch.bool)
            for lx, lz in local:
                px, pz = knee_x + c * lx - s * lz, knee_z + s * lx + c * lz
                touching |= (pz <= self._height(px[:, None])[:, 0] + CONTACT_TOLERANCE)
            contacts.append(touching.float())
        return contacts

    def _lidar(self, x, z):
        t = self.world_terrain
        start = self.bank.seg_start[t, self._cell(x)]
        index = (start[:, None] + self._lidar_slots).clamp(max=self.bank.segments.shape[1] - 1)
        seg = self.bank.segments[t[:, None], index]
        ax, ay = seg[..., 0], seg[..., 1]
        sx, sy = seg[..., 2] - ax, seg[..., 3] - ay
        dx, dy = (ax - x[:, None])[:, None], (ay - z[:, None])[:, None]
        rx, ry = self._rays[None, :, 0, None], self._rays[None, :, 1, None]
        sx, sy = sx[:, None], sy[:, None]
        denominator = rx * sy - ry * sx
        safe = torch.where(denominator.abs() > 1e-9, denominator, torch.full_like(denominator, 1e-9))
        along_ray = (dx * sy - dy * sx) / safe
        along_segment = (dx * ry - dy * rx) / safe
        hit = (denominator.abs() > 1e-9) & (along_ray >= 0) & (along_ray <= 1) & (along_segment >= 0) & (along_segment <= 1)
        return torch.where(hit, along_ray, torch.ones_like(along_ray)).min(dim=-1).values

    def _get_observations(self):
        x, z, angle, legs = self._pose()
        qd = self.robot.data.joint_vel.torch
        leg_speed = -qd[:, self._leg_ids] / self._speed_limits
        left, right = self._foot_contacts(x, z, angle, legs)
        obs = torch.stack((
            angle, 2.0 * -qd[:, self._root_pitch] / spec.FPS,
            0.3 * qd[:, self._root_x] * (spec.VIEWPORT_W / spec.SCALE) / spec.FPS,
            0.3 * qd[:, self._root_z] * (spec.VIEWPORT_H / spec.SCALE) / spec.FPS,
            legs[:, 0], leg_speed[:, 0], legs[:, 1] + 1.0, leg_speed[:, 1], left,
            legs[:, 2], leg_speed[:, 2], legs[:, 3] + 1.0, leg_speed[:, 3], right,
        ), dim=-1)
        return {"policy": torch.cat((obs, self._lidar(x, z)), dim=-1)}

    def _get_dones(self):
        x, z, angle, _ = self._pose()
        q, qd = self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch
        finite = torch.isfinite(q).all(dim=-1) & torch.isfinite(qd).all(dim=-1)
        c, s = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
        hx, hz = self._hull_points[:, 0], self._hull_points[:, 1]
        px, pz = x[:, None] + c * hx - s * hz, z[:, None] + s * hx + c * hz
        touching = (pz <= self._height(px)).any(dim=-1)
        self._numerical_failure = ~finite
        self.fallen = touching | (x < 0.0) | ~finite
        self.finished = (x > spec.FINISH_X) & ~self.fallen
        return self.fallen | self.finished, self.episode_length_buf >= self.max_episode_length

    def _get_rewards(self):
        x, _, angle, _ = self._pose()
        shaping = 130.0 * x / spec.SCALE - 5.0 * angle.abs()
        reward = shaping - self.prev_shaping - 0.00035 * spec.MOTORS_TORQUE * self.actions.abs().sum(dim=-1)
        self.prev_shaping = shaping
        return torch.where(self.fallen, torch.full_like(reward, -100.0), reward)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        if self.assigner is not None:
            self.world_terrain[env_ids] = self.assigner(env_ids)
        count = len(env_ids)
        q = self.robot.data.default_joint_pos.torch[env_ids].clone()
        qd = torch.zeros_like(q)
        force = self.bank.initial_force[self.world_terrain[env_ids]]
        qd[:, self._root_x] = spec.START_HULL[3] + force * spec.INITIAL_FORCE_TO_VX
        qd[:, self._root_z] = spec.START_HULL[4]
        qd[:, self._root_pitch] = -spec.START_HULL[5]
        speeds = torch.tensor((spec.START_HIP_SPEED[0], spec.START_KNEE_SPEED[0],
                               spec.START_HIP_SPEED[1], spec.START_KNEE_SPEED[1]), device=self.device)
        qd[:, self._leg_ids] = -speeds.expand(count, 4)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
        self.actions[env_ids] = 0.0
        self.prev_shaping[env_ids] = 130.0 * spec.START_HULL[0] / spec.SCALE - 5.0 * abs(spec.START_HULL[2])

    def reset_worlds(self, env_ids):
        """Reset worlds outside step(), for example to start newly queued work."""
        self._reset_idx(env_ids)
        self.episode_length_buf[env_ids] = 0
