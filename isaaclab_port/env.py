"""Local, planar POET task for Isaac Lab 3.x.

This is a deliberately disclosed simulator/task port, not the paper's Box2D task.
The 24 policy inputs are pitch, scaled pitch/x/z velocities, four leg angles,
four scaled leg velocities, two foot-frame heights above terrain, and ten known
terrain heights relative to the hip. The last twelve inputs replace the paper's
contact flags and lidar with geometric information. Distances are metres, angles
radians, velocities m/s or rad/s, and actions are normalized joint torques.

Import this module in an Isaac Lab environment; start the selected runtime with
``isaaclab.app.launch_simulation(cfg, args)`` before constructing the environment.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UrdfFileCfg, UsdFileCfg
from isaaclab.utils.configclass import configclass

ASSET_PATH = Path(__file__).resolve().parent / "assets" / "planar_biped.urdf"
USD_ASSET_PATH = ASSET_PATH.with_suffix(".usda")
LEG_JOINTS = ("left_hip", "left_knee", "right_hip", "right_knee")
ROOT_JOINTS = ("root_x", "root_z", "root_pitch")


def build_course_profile(
    terrain: Mapping[str, float], seed: int, *, course_length: float = 30.0,
    tile_length: float = 0.4, flat_margin: float = 4.0,
) -> np.ndarray:
    """Return static tile-top heights; NaN means an actual missing floor tile.

    Only the named geometric parameters affect the profile; a possible ``seed``
    metadata entry in ``terrain`` is accepted but the explicit seed is used.
    Generation uses a private RNG, so querying terrain never changes ES noise.
    """
    values = {key: float(terrain.get(key, 0.0)) for key in ("roughness", "stump_height", "gap_width")}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("Terrain roughness, stump_height and gap_width must be finite and nonnegative")
    if tile_length <= 0.0 or flat_margin < 0.0 or course_length <= 2.0 * flat_margin:
        raise ValueError("Require tile_length > 0 and course_length > 2 * flat_margin >= 0")
    tile_count = round(course_length / tile_length)
    if not math.isclose(tile_count * tile_length, course_length, abs_tol=1e-6):
        raise ValueError("course_length must be an integer multiple of tile_length")
    rng = np.random.default_rng(int(seed))
    heights = np.zeros(tile_count, dtype=np.float32)
    margin = math.ceil(flat_margin / tile_length)
    first, last = margin, tile_count - margin
    roughness = values["roughness"]
    height = 0.0
    for index in range(first, last):
        height = float(np.clip(height + rng.uniform(-0.25, 0.25) * roughness, 0.0, roughness))
        heights[index] = height

    # Separate periodic candidate locations keep stumps and pits interpretable.
    # Widths are quantized to 0.4 m tiles and documented in the saved profile.
    if values["stump_height"] > 0.0:
        for index in range(first + 4, last - 1, 17):
            heights[index:index + 2] += rng.uniform(0.5, 1.0) * values["stump_height"]
    if values["gap_width"] > 0.0:
        gap_tiles = max(1, math.ceil(values["gap_width"] / tile_length))
        for index in range(first + 11, last, max(23, gap_tiles + 8)):
            heights[index:min(index + gap_tiles, last)] = np.nan
    return heights


@configclass
class PoetBipedEnvCfg(DirectRLEnvCfg):
    """Configuration for one terrain replicated across ES candidate lanes."""

    decimation = 2
    episode_length_s = 2000.0 / 60.0
    action_space = 4
    observation_space = 24
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # Newton assigns separate world IDs, so physical worlds may share the
        # same coordinates. This avoids lane-dependent float32 contact errors
        # from translating the long course far from the origin. Replay one env.
        num_envs=32, env_spacing=0.0, replicate_physics=True, clone_in_fabric=False,
    )
    # The port uses no cameras or network-hosted visual assets.
    ui_window_class_type = None
    video_recorder = None
    terrain: dict = {"roughness": 0.0, "stump_height": 0.0, "gap_width": 0.0}
    terrain_seed: int = 0
    course_length: float = 30.0
    tile_length: float = 0.4
    lane_width: float = 2.0
    flat_margin: float = 4.0
    start_x: float = 2.0
    start_height: float = 0.98
    torque_scale: float = 40.0
    fall_height: float = 0.35
    fall_pitch: float = 1.3
    # A complete 28 m traverse yields 300 progress points before penalties.
    progress_return: float = 300.0
    attitude_penalty: float = 0.01
    effort_penalty: float = 0.00005
    robot_cfg: ArticulationCfg | None = None


def make_env_cfg(
    terrain: Mapping[str, float], *, seed: int | None = None, num_envs: int = 32,
    device: str = "cuda:0", urdf_path: str | Path | None = None,
    physics=None, episode_steps: int = 2000,
) -> PoetBipedEnvCfg:
    """Build an explicit config, using Newton unless a backend cfg is supplied."""
    if num_envs < 1 or episode_steps < 1:
        raise ValueError("num_envs and episode_steps must be positive")
    # The self-contained USD is the default: it needs neither Kit nor a network
    # asset download. An explicitly supplied URDF still uses the official importer.
    asset_path = Path(urdf_path or (USD_ASSET_PATH if USD_ASSET_PATH.is_file() else ASSET_PATH)).resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)
    cfg = PoetBipedEnvCfg()
    cfg.terrain = {key: float(terrain.get(key, 0.0)) for key in ("roughness", "stump_height", "gap_width")}
    cfg.terrain_seed = int(terrain.get("seed", 0) if seed is None else seed)
    cfg.seed = cfg.terrain_seed
    cfg.scene.num_envs = int(num_envs)
    cfg.sim.device = device
    cfg.episode_length_s = math.nextafter(int(episode_steps) * cfg.decimation * cfg.sim.dt, 0.0)
    if physics is None:
        from isaaclab_newton.physics import NewtonCfg
        physics = NewtonCfg()
    cfg.sim.physics = physics
    if asset_path.suffix.lower() == ".urdf":
        spawn_cfg = UrdfFileCfg(
            asset_path=str(asset_path), fix_base=True, merge_fixed_joints=False,
            self_collision=False,
            joint_drive=UrdfFileCfg.JointDriveCfg(target_type="none"),
        )
    elif asset_path.suffix.lower() in {".usd", ".usda", ".usdc"}:
        spawn_cfg = UsdFileCfg(usd_path=str(asset_path))
    else:
        raise ValueError(f"Expected a USD or URDF robot asset, got {asset_path}")
    cfg.robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=spawn_cfg,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={"root_x": cfg.start_x, "root_z": cfg.start_height, "root_pitch": 0.0,
                       "left_hip": 0.0, "left_knee": 0.0, "right_hip": 0.0, "right_knee": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=list(LEG_JOINTS), stiffness=0.0, damping=0.0,
                effort_limit=cfg.torque_scale, effort_limit_sim=cfg.torque_scale,
            ),
        },
    )
    # Validate geometry now, before the runtime or physics stage is created.
    build_course_profile(cfg.terrain, cfg.terrain_seed, course_length=cfg.course_length,
                         tile_length=cfg.tile_length, flat_margin=cfg.flat_margin)
    return cfg


class PoetBipedEnv(DirectRLEnv):
    """Independent, identical terrain lanes; one four-torque action per lane.

    ``terminal_success`` is a per-step Boolean tensor captured before automatic
    reset. It survives ``_reset_idx`` so the evaluator can score the first episode
    without accidentally reading the reset state as its terminal state.
    """

    cfg: PoetBipedEnvCfg

    def __init__(self, cfg: PoetBipedEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.robot_cfg is None:
            raise ValueError("Construct the task config with make_env_cfg()")
        self.course_heights = build_course_profile(
            cfg.terrain, cfg.terrain_seed, course_length=cfg.course_length,
            tile_length=cfg.tile_length, flat_margin=cfg.flat_margin,
        )
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._root_ids, root_names = self.robot.find_joints(list(ROOT_JOINTS), preserve_order=True)
        self._leg_ids, leg_names = self.robot.find_joints(list(LEG_JOINTS), preserve_order=True)
        self._foot_ids, foot_names = self.robot.find_bodies(["left_foot", "right_foot"], preserve_order=True)
        if root_names != list(ROOT_JOINTS) or leg_names != list(LEG_JOINTS) or len(foot_names) != 2:
            raise RuntimeError(f"Unexpected imported robot names: {root_names}, {leg_names}, {foot_names}")
        self._root_x, self._root_z, self._root_pitch = self._root_ids
        # A pit is observed as ground 2 m below the nominal surface; no collision
        # geometry is placed there. This is privileged terrain preview, not lidar.
        heights = np.nan_to_num(self.course_heights, nan=-2.0)
        self._height_table = torch.as_tensor(heights, dtype=torch.float32, device=self.device)
        self._preview_offsets = torch.linspace(0.4, 4.0, 10, device=self.device)
        self.actions = torch.zeros((self.num_envs, 4), device=self.device)
        self._previous_x = torch.full((self.num_envs,), cfg.start_x, device=self.device)
        self._fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminal_success = torch.zeros_like(self._fallen)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        stage = self.sim.stage
        terrain_path = "/World/envs/env_0/Course"
        UsdGeom.Xform.Define(stage, terrain_path)
        material = UsdShade.Material.Define(stage, terrain_path + "/GroundMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(0.8)
        physics_material.CreateDynamicFrictionAttr(0.8)
        physics_material.CreateRestitutionAttr(0.0)
        thickness = 2.0
        for index, top in enumerate(self.course_heights):
            if not np.isfinite(top):
                continue
            cube = UsdGeom.Cube.Define(stage, f"{terrain_path}/Tile_{index:03d}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(
                (index + 0.5) * self.cfg.tile_length, 0.0, float(top) - thickness / 2.0,
            ))
            cube.AddScaleOp().Set(Gf.Vec3f(self.cfg.tile_length, self.cfg.lane_width, thickness))
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.34, 0.39, 0.28)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose="physics")
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        if str(self.device).startswith("cpu"):
            self.scene.filter_collisions(global_prim_paths=[])

    def _terrain_height(self, positions_x: torch.Tensor) -> torch.Tensor:
        indices = torch.floor(positions_x / self.cfg.tile_length).long()
        indices = indices.clamp(0, self._height_table.numel() - 1)
        return self._height_table[indices]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target_index(
            target=self.cfg.torque_scale * self.actions, joint_ids=self._leg_ids,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        q = self.robot.data.joint_pos.torch
        qd = self.robot.data.joint_vel.torch
        x, z = q[:, self._root_x], q[:, self._root_z]
        root_obs = torch.stack((q[:, self._root_pitch], qd[:, self._root_pitch] * 0.1,
                                qd[:, self._root_x] * 0.2, qd[:, self._root_z] * 0.2), dim=-1)
        foot_positions = self.robot.data.body_link_pos_w.torch[:, self._foot_ids]
        # The fixed-foot frames sit at ankle height, not the contact patch.
        foot_x = foot_positions[..., 0] - self.scene.env_origins[:, None, 0]
        foot_z = foot_positions[..., 2] - self.scene.env_origins[:, None, 2]
        foot_heights = foot_z - self._terrain_height(foot_x)
        preview = self._terrain_height(x[:, None] + self._preview_offsets) - z[:, None]
        obs = torch.cat((root_obs, q[:, self._leg_ids], qd[:, self._leg_ids] * 0.1,
                         foot_heights.clamp(-2.0, 2.0), preview.clamp(-3.0, 3.0)), dim=-1)
        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.robot.data.joint_pos.torch
        qd = self.robot.data.joint_vel.torch
        finite = torch.isfinite(q).all(dim=-1) & torch.isfinite(qd).all(dim=-1)
        self._fallen = (q[:, self._root_z] < self.cfg.fall_height) | (
            q[:, self._root_pitch].abs() > self.cfg.fall_pitch
        ) | (q[:, self._root_x] < 0.0) | ~finite
        self.terminal_success = (q[:, self._root_x] >= self.cfg.course_length) & ~self._fallen
        self.extras["terminal_success"] = self.terminal_success.clone()
        self.extras["numerical_failure"] = (~finite).clone()
        terminated = self._fallen | self.terminal_success
        # DirectRLEnv increments the counter before calling this hook.
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        q = self.robot.data.joint_pos.torch
        x = q[:, self._root_x].clamp(max=self.cfg.course_length)
        progress_scale = self.cfg.progress_return / (self.cfg.course_length - self.cfg.start_x)
        reward = progress_scale * (x - self._previous_x)
        reward -= self.cfg.attitude_penalty * q[:, self._root_pitch].abs()
        reward -= self.cfg.effort_penalty * self.cfg.torque_scale * self.actions.abs().sum(dim=-1)
        self._previous_x.copy_(x)
        return torch.where(self._fallen, torch.full_like(reward, -100.0), reward)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        q = self.robot.data.default_joint_pos.torch[env_ids].clone()
        qd = self.robot.data.default_joint_vel.torch[env_ids].clone()
        # The base anchor stays fixed at each lane origin. Reset only its seven
        # generalized coordinates; moving a fixed articulation root is unnecessary.
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
        self.actions[env_ids] = 0.0
        self._previous_x[env_ids] = self.cfg.start_x
        # terminal_success intentionally retains the just-finished step's flags.
