"""Read-only diagnostics for the requested uv environment and GA source version."""
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys

GA_COMMIT = "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4"
GA_PHYSICS_PACKAGES = {"torch": "2.11.0", "warp-lang": "1.17.0", "newton": "1.6.1"}


def inspect():
    report = {"python": sys.version, "executable": sys.executable,
              "target_isaaclab": "v3.0.0", "target_commit": GA_COMMIT, "packages": {}}
    for name in ("isaaclab", "isaaclab-assets", "isaaclab-tasks", "isaaclab-newton",
                 "isaacsim", "torch", "warp-lang", "newton", "gymnasium", "numpy"):
        try:
            report["packages"][name] = md.version(name)
        except md.PackageNotFoundError:
            report["packages"][name] = None
    try:
        distribution = md.distribution("isaaclab")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        report["source_url"] = direct.get("url")
        if direct.get("url", "").startswith("file://"):
            source = Path(direct["url"][7:])
            git = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
            report["source_commit"] = git.stdout.strip() if git.returncode == 0 else None
            status = subprocess.run(["git", "-C", str(source), "status", "--porcelain"], capture_output=True, text=True)
            report["source_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (md.PackageNotFoundError, ValueError, OSError):
        pass
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        report["cuda_available"] = False
        report["cuda_device_count"] = 0
    report["ga_source_confirmed"] = (
        report.get("source_commit") == GA_COMMIT and report.get("source_dirty") is False
    )
    report["ga_dependency_mismatches"] = {
        name: {"expected": expected, "installed": report["packages"].get(name)}
        for name, expected in GA_PHYSICS_PACKAGES.items()
        if (report["packages"].get(name) or "").split("+")[0] != expected
    }
    report["ga_environment_confirmed"] = (
        report["ga_source_confirmed"] and not report["ga_dependency_mismatches"]
        and sys.version_info[:2] == (3, 12)
    )
    return report


if __name__ == "__main__":
    print(json.dumps(inspect(), indent=2))
