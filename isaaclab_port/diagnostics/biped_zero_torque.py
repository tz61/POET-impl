"""Record actual biped dynamics without termination-triggered resets.

The stepping loop mirrors DirectRLEnv's physics section. Actions are zero;
termination flags are observed but never reset the robot, preserving evidence
of the first physical failure. No simulator/environment packages are modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("featherstone", "mujoco_cpu"), default="featherstone")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--actions", type=int, default=20)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = dict(solver=args.solver, substeps=args.substeps, actions=args.actions,
                  ok=False, phase="imports", records=[])
    env = None
    try:
        import numpy as np
        import torch
        import warp as wp
        try:
            from isaaclab.app import launch_simulation
        except ImportError:
            from isaaclab_tasks.utils import launch_simulation
        from isaaclab_newton.physics import NewtonCfg, FeatherstoneSolverCfg, MJWarpSolverCfg
        from isaaclab_port.core import Terrain
        from isaaclab_port.env import make_env_cfg, PoetBipedEnv
        from isaaclab_port.doctor import inspect
        wp.config.kernel_cache_dir = str(ROOT / ".cache" / "warp")
        result["environment"] = inspect()
        solver = (FeatherstoneSolverCfg() if args.solver == "featherstone"
                  else MJWarpSolverCfg(use_mujoco_cpu=True))
        cfg = make_env_cfg(asdict(Terrain(seed=24582922)), seed=24582922,
            num_envs=1, device="cpu", episode_steps=args.actions + 1,
            physics=NewtonCfg(solver_cfg=solver, use_cuda_graph=False,
                              num_substeps=args.substeps))
        result["dt"] = cfg.sim.dt
        result["decimation"] = cfg.decimation
        result["phase"] = "construct_environment"
        with launch_simulation(cfg):
            try:
                env = PoetBipedEnv(cfg)
                env.reset(seed=24582922)
                result["joint_names"] = list(env.robot.joint_names)
                result["body_names"] = list(env.robot.body_names)
                result["physics_handles_decimation"] = env._physics_handles_decimation

                def record(action_index):
                    q = env.robot.data.joint_pos.torch.detach().cpu().numpy().copy()
                    qd = env.robot.data.joint_vel.torch.detach().cpu().numpy().copy()
                    pos = env.robot.data.body_link_pos_w.torch.detach().cpu().numpy().copy()
                    vel = env.robot.data.body_link_vel_w.torch.detach().cpu().numpy().copy()
                    terminated, truncated = env._get_dones()
                    row = dict(action=action_index, seconds=action_index * env.step_dt,
                               q=q.tolist(), qd=qd.tolist(), body_pos=pos.tolist(),
                               body_vel=vel.tolist(), terminated=terminated.cpu().tolist(),
                               finite=bool(np.isfinite(q).all() and np.isfinite(qd).all()
                                           and np.isfinite(pos).all() and np.isfinite(vel).all()))
                    result["records"].append(row)
                    print(json.dumps(row), flush=True)
                    return row

                result["phase"] = "zero_torque_steps"
                record(0)
                zero = torch.zeros((1, 4), device=env.device)
                with torch.no_grad():
                    for action_index in range(1, args.actions + 1):
                        env._pre_physics_step(zero)
                        if env._physics_handles_decimation:
                            env._sim_step_counter += cfg.decimation
                            env._apply_action()
                            env.scene.write_data_to_sim()
                            env.sim.step(render=False)
                            env.scene.update(dt=env.step_dt)
                        else:
                            for _ in range(cfg.decimation):
                                env._sim_step_counter += 1
                                env._apply_action()
                                env.scene.write_data_to_sim()
                                env.sim.step(render=False)
                                env.scene.update(dt=env.physics_dt)
                        env.episode_length_buf += 1
                        env.common_step_counter += 1
                        row = record(action_index)
                        if not row["finite"]:
                            break
                finite = all(row["finite"] for row in result["records"])
                max_abs_qd = max(float(np.max(np.abs(row["qd"])))
                                 for row in result["records"] if row["finite"])
                max_abs_root_x = max(abs(float(row["q"][0][0]) - cfg.start_x)
                                     for row in result["records"] if row["finite"])
                max_root_z = max(float(row["q"][0][1])
                                 for row in result["records"] if row["finite"])
                bounded = max_abs_qd < 50.0 and max_abs_root_x < 2.0 and max_root_z < 2.0
                result.update(diagnostic_completed=True, all_states_finite=finite,
                              max_abs_generalized_velocity=max_abs_qd,
                              max_root_x_displacement=max_abs_root_x,
                              max_root_z=max_root_z,
                              bounded_zero_torque_motion=bounded,
                              ok=finite and bounded, phase="complete")
            finally:
                if env is not None:
                    env.close()
                    env = None
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        args.result.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
