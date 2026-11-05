#!/usr/bin/env python3
"""Record actual first-episode link poses from a frozen fixed-task checkpoint.

The output supports a clearly labeled reconstruction of simulator states.
It is not a captured viewport video and never changes the trained policy.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "poet-reproduction"
sys.path.insert(0, str(PROJECT))
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT / ".cache"))
os.environ.setdefault("WARP_CACHE_PATH", str(PROJECT / ".cache" / "warp"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def visual_geometry():
    """Read the real primitive asset's local box geometry for rendering."""
    path = PROJECT / "isaaclab_port" / "assets" / "planar_biped.urdf"
    robot = ET.parse(path).getroot()
    materials = {item.attrib["name"]: [float(x) for x in item.find("color").attrib["rgba"].split()]
                 for item in robot.findall("material")}
    geometry = {}
    for link in robot.findall("link"):
        boxes = []
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            boxes.append({"local_xyz": [float(x) for x in origin.attrib["xyz"].split()],
                          "local_rpy": [float(x) for x in origin.attrib.get("rpy", "0 0 0").split()],
                          "size_xyz": [float(x) for x in visual.find("geometry/box").attrib["size"].split()],
                          "rgba": materials[visual.find("material").attrib["name"]]})
        geometry[link.attrib["name"]] = boxes
    return geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-index", type=int, default=0,
                        help="Current policy by fixed training-task index; 0 is the flat task")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--heldout-task-index", type=int, default=None,
                        help="Use this index in the run's frozen evaluation bank instead of a training task")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new file to preserve prior artifacts")
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    import numpy as np
    import torch
    import warp as wp
    from isaaclab_port.core import Terrain
    from isaaclab_port.doctor import inspect
    from isaaclab_port.env import PoetBipedEnv, make_env_cfg
    from isaaclab_port.runtime import Evaluator, policy_actions
    from isaaclab_newton.physics import NewtonCfg, FeatherstoneSolverCfg
    try:
        from isaaclab.app import launch_simulation
    except ImportError:
        from isaaclab_tasks.utils import launch_simulation

    wp.config.kernel_cache_dir = str(PROJECT / ".cache" / "warp")
    checkpoint = json.loads(args.checkpoint.read_text())
    fingerprint = json.loads(args.checkpoint.with_name("task_fingerprint.json").read_text())
    payload = fingerprint["payload"]
    source_files = {"task": PROJECT / "isaaclab_port" / "env.py",
                    "evaluator": PROJECT / "isaaclab_port" / "runtime.py",
                    "algorithm": PROJECT / "isaaclab_port" / "core.py",
                    "robot_usd": PROJECT / "isaaclab_port" / "assets" / "planar_biped.usda"}
    if any(payload["source_sha256"][name] != sha(path) for name, path in source_files.items()):
        raise ValueError("Current task/algorithm/robot source differs from this checkpoint")
    environment = inspect()
    versions = {"source_commit": environment.get("source_commit"), **environment["packages"]}
    if versions != payload["simulation_versions"]:
        raise ValueError("Current physics environment differs from this checkpoint")
    if payload["solver"] != "featherstone" or payload["task_config"]["device_family"] != "cpu":
        raise ValueError("This recorder supports the exact CPU/Featherstone comparison only")
    if not 0 <= args.policy_index < len(checkpoint["pairs"]) or not 0 <= args.task_index < len(checkpoint["pairs"]):
        parser.error("policy-index/task-index outside checkpoint population")
    theta_array = np.array(checkpoint["pairs"][args.policy_index]["theta"], dtype=np.float64)
    task = checkpoint["pairs"][args.task_index]["terrain"]
    task_origin = "training"
    task_id = f"training_{args.task_index}"
    if args.heldout_task_index is not None:
        protocol = json.loads(args.checkpoint.parent.parent.joinpath("protocol.json").read_text())
        if not 0 <= args.heldout_task_index < len(protocol["heldout_tasks"]):
            parser.error("heldout-task-index outside the frozen bank")
        selected_task = protocol["heldout_tasks"][args.heldout_task_index]
        task = selected_task["terrain"]
        task_origin, task_id = "heldout", selected_task["task_id"]
    cfg = make_env_cfg(task, seed=task["seed"], num_envs=1, device="cpu",
                      episode_steps=payload["episode_steps"],
                      physics=NewtonCfg(solver_cfg=FeatherstoneSolverCfg(), use_cuda_graph=False,
                                        num_substeps=payload["substeps"]))

    class RecordingEnv(PoetBipedEnv):
        def frame(self):
            data = self.robot.data
            values = {"body_positions_xyz": data.body_link_pos_w.torch[0].detach().cpu().tolist(),
                      "body_quaternions_xyzw": data.body_link_quat_w.torch[0].detach().cpu().tolist(),
                      "joint_positions": data.joint_pos.torch[0].detach().cpu().tolist(),
                      "joint_velocities": data.joint_vel.torch[0].detach().cpu().tolist()}
            if any(not np.isfinite(value).all() for value in values.values()):
                raise FloatingPointError("Nonfinite replay state")
            return values

        def _get_dones(self):
            result = super()._get_dones()
            self.recorded_frame = self.frame()  # before automatic reset
            return result

    result = {"evidence_kind": "actual_simulator_state_reconstruction", "is_viewport_capture": False,
              "description": "Recorded link poses at control steps; render as a labeled reconstruction of real simulation, not viewport footage.",
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha(args.checkpoint),
              "source_task_fingerprint": fingerprint, "checkpoint_iteration": checkpoint["iteration"],
              "source_mode": (checkpoint["settings"].get("periodic_transfer_mode", "full")
                              if checkpoint["settings"]["transfers"] else "no_transfer"),
              "source_settings": checkpoint["settings"], "policy_index": args.policy_index,
              "task_index": args.heldout_task_index if task_origin == "heldout" else args.task_index,
              "task_origin": task_origin, "task_id": task_id, "terrain": task, "environment": environment,
              "policy_sha256": hashlib.sha256(theta_array.astype("<f8").tobytes()).hexdigest(),
              "replay_num_envs": 1, "visual_geometry": visual_geometry(),
              "recorder_sha256": sha(__file__),
              "visual_geometry_urdf_sha256": sha(PROJECT / "isaaclab_port" / "assets" / "planar_biped.urdf"),
              "frames": []}
    env = None
    with launch_simulation(cfg):
        try:
            env = RecordingEnv(cfg)
            obs, _ = env.reset(seed=task["seed"])
            control_dt = cfg.sim.dt * cfg.decimation
            result.update(body_names=list(env.robot.body_names), joint_names=list(env.robot.joint_names),
                          control_dt=control_dt, quaternion_convention="xyzw",
                          terrain_profile=[None if np.isnan(x) else float(x) for x in env.course_heights],
                          tile_length=cfg.tile_length, lane_width=cfg.lane_width,
                          source_episode_steps=payload["episode_steps"])
            result["frames"].append({"step": 0, "time": 0., "reward": None, **env.frame()})
            theta = torch.as_tensor(theta_array[None], dtype=torch.float32, device="cpu")
            total = 0.
            finished = success = False
            with torch.no_grad():
                for step in range(1, min(args.max_steps, env.max_episode_length) + 1):
                    action = policy_actions(obs["policy"], theta)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if info.get("numerical_failure", torch.zeros(1, dtype=torch.bool))[0]:
                        raise FloatingPointError("Numerical failure during replay")
                    total += float(reward[0])
                    finished = bool(terminated[0] or truncated[0])
                    success = bool(info["terminal_success"][0])
                    result["frames"].append({"step": step, "time": step * control_dt,
                        "reward": float(reward[0]), "actions": action[0].cpu().tolist(),
                        "terminated": bool(terminated[0]), "truncated": bool(truncated[0]),
                        "success": success, **env.recorded_frame})
                    if finished:
                        break
            result.update(return_=total, control_steps=step, first_episode_finished=finished,
                          reached_goal=success, stop_reason="episode_done" if finished else "recording_cap")
            root_x_index = result["joint_names"].index("root_x")
            result["terminal_x"] = result["frames"][-1]["joint_positions"][root_x_index]
            result["max_x"] = max(frame["joint_positions"][root_x_index] for frame in result["frames"])
            if task_origin == "training" and args.policy_index == args.task_index and finished:
                expected = checkpoint["pairs"][args.policy_index]["score"]
                result["checkpoint_own_score_verification"] = {
                    "expected": expected, "observed": total,
                    "passed": bool(np.isclose(expected, total, atol=1e-6, rtol=1e-8))}
        finally:
            if env is not None:
                env.close()
        if result["first_episode_finished"]:
            verifier = Evaluator(1, lambda terrain: cfg)
            try:
                verified_return = float(verifier(Terrain(**task), theta_array[None])[0])
                verification = {"return": verified_return,
                                "length": verifier.last_episode_lengths[0],
                                "success": verifier.last_episode_successes[0],
                                "return_difference": verified_return - result["return_"]}
                verification["passed"] = bool(
                    np.isclose(verified_return, result["return_"], atol=1e-6, rtol=1e-8)
                    and verification["length"] == result["control_steps"]
                    and verification["success"] == result["reached_goal"])
                result["production_evaluator_verification"] = verification
            finally:
                verifier.close()
        else:
            result["production_evaluator_verification"] = {"passed": False, "reason": "recording cap before episode end"}
    evaluation_path = args.checkpoint.with_name(f"evaluation_{checkpoint['iteration']:03d}.json")
    if task_origin == "heldout" and result["first_episode_finished"] and evaluation_path.exists():
        saved = json.loads(evaluation_path.read_text())["evaluation"]["tasks"][args.heldout_task_index]
        if saved["terrain"] != task:
            raise ValueError("Saved evaluation task differs from the replay task")
        index = args.policy_index
        expected = {"return": saved["returns_by_policy"][index],
                    "length": saved["episode_lengths_by_policy"][index],
                    "success": saved["goal_successes_by_policy"][index],
                    "max_x": saved["max_x_by_policy"][index],
                    "terminal_x": saved["terminal_x_by_policy"][index]}
        result["source_evaluation_verification"] = {
            "evaluation_file": str(evaluation_path.resolve()), "sha256": sha(evaluation_path),
            "source_num_envs": payload["task_config"]["num_envs"], "expected": expected,
            "passed": bool(np.isclose(expected["return"], result["return_"], atol=1e-6, rtol=1e-8)
                           and expected["length"] == result["control_steps"]
                           and expected["success"] == result["reached_goal"]
                           and np.isclose(expected["max_x"], result["max_x"], atol=1e-6)
                           and np.isclose(expected["terminal_x"], result["terminal_x"], atol=1e-6))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "frames": len(result["frames"]),
                      "return": result["return_"], "success": result["reached_goal"],
                      "verification": result["production_evaluator_verification"]}), flush=True)
    if result["first_episode_finished"] and not result["production_evaluator_verification"]["passed"]:
        raise SystemExit("Replay/evaluator verification mismatch; retained JSON for diagnosis")
    if result.get("source_evaluation_verification", {}).get("passed") is False:
        raise SystemExit("Replay/source snapshot verification mismatch; retained JSON for diagnosis")
    if result.get("checkpoint_own_score_verification", {}).get("passed") is False:
        raise SystemExit("Replay/checkpoint own-task score mismatch; retained JSON for diagnosis")


if __name__ == "__main__":
    main()
