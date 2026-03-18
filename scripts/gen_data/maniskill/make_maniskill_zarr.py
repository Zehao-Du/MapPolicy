from __future__ import annotations

import argparse
import glob
import pathlib
import shlex
import subprocess
import sys


def run_cmd(cmd: list[str], dry_run: bool = False):
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n[RUN] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def resolve_replay_output_path(
    motionplanning_dir: pathlib.Path,
    traj_name: str,
    obs_mode: str,
    control_mode: str,
) -> pathlib.Path:
    exact = motionplanning_dir / f"{traj_name}.{obs_mode}.{control_mode}.physx_cpu.h5"
    if exact.exists():
        return exact

    pattern = str(motionplanning_dir / f"{traj_name}.{obs_mode}.{control_mode}.*.h5")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        return pathlib.Path(candidates[-1])

    raise FileNotFoundError(
        f"未找到 replay 输出文件。\n"
        f"期望路径: {exact}\n"
        f"或匹配: {pattern}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="一键生成 ManiSkill zarr：采集h5 -> replay点云h5 -> 转zarr"
    )
    parser.add_argument("--task-name", required=True, help="任务名，例如 PickCube-v1")
    parser.add_argument("--num-traj", type=int, required=True, help="轨迹数量，例如 100")
    parser.add_argument(
        "--control-mode",
        required=True,
        help="控制模式，例如 pd_ee_delta_pose 或 pd_joint_pos",
    )
    parser.add_argument(
        "--obs-mode",
        default="rgb+depth+segmentation",
        help="replay 的观测模式，例如 rgb+depth+segmentation 或 pointcloud",
    )

    parser.add_argument(
        "--traj-name",
        default=None,
        help="轨迹名前缀（默认取 task-name 去掉 -v1 等后缀）",
    )
    parser.add_argument(
        "--record-dir",
        default="maniskill/data",
        help="原始轨迹保存根目录（run/replay 用）",
    )
    parser.add_argument(
        "--zarr-save-dir",
        default="/data2/lirui/StructureMap3D/data_new/maniskill_zarr",
        help="zarr 保存根目录",
    )
    parser.add_argument(
        "--camera-name",
        default="base_camera",
        help="传给 h52zarr.py 的 camera_name",
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="replay 时允许失败轨迹（默认不允许）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印命令，不实际执行",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parent
    traj_name = args.traj_name or args.task_name.split("-")[0]

    motionplanning_dir = repo_root / args.record_dir / args.task_name / "motionplanning"
    raw_h5_path = motionplanning_dir / f"{traj_name}.h5"
    zarr_out_dir = pathlib.Path(args.zarr_save_dir) / args.task_name

    run_cmd(
        [
            sys.executable,
            "-m",
            "mani_skill.examples.motionplanning.panda.run",
            "--env-id",
            args.task_name,
            "--num-traj",
            str(args.num_traj),
            "--only-count-success",
            "--save-video",
            "--record-dir",
            str(repo_root / args.record_dir),
            "--traj-name",
            traj_name,
        ],
        dry_run=args.dry_run,
    )

    replay_cmd = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(raw_h5_path),
        "-c",
        args.control_mode,
        "-o",
        args.obs_mode,
        "--no-vis",
        "--save-traj",
        "--verbose",
    ]
    if not args.allow_failure:
        replay_cmd.append("--no-allow-failure")
    run_cmd(replay_cmd, dry_run=args.dry_run)

    replay_h5_path = resolve_replay_output_path(
        motionplanning_dir=motionplanning_dir,
        traj_name=traj_name,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
    )

    run_cmd(
        [
            sys.executable,
            str(repo_root / "h52zarr.py"),
            "--input-path",
            str(replay_h5_path),
            "--zarr-save-dir",
            str(zarr_out_dir),
            "--camera-name",
            args.camera_name,
        ],
        dry_run=args.dry_run,
    )

    print("\n[OK] 全流程完成")
    print(f"  task_name      : {args.task_name}")
    print(f"  num_traj       : {args.num_traj}")
    print(f"  control_mode   : {args.control_mode}")
    print(f"  raw_h5         : {raw_h5_path}")
    print(f"  replay_h5      : {replay_h5_path}")
    print(f"  zarr_save_dir  : {zarr_out_dir}")


if __name__ == "__main__":
    main()