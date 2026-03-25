生成原始.h5数据

python -m mani_skill.examples.motionplanning.panda.run --env-id "StackCube-v1" --num-traj 1100 --only-count-success --save-video --record-dir Data/maniskill_1024 --traj-name StackCube --num-procs 10

replay数据并获取点云，更改控制方式

CUDA_VISIBLE_DEVICES=0 python -m mani_skill.trajectory.replay_trajectory \
--traj-path Data/maniskill_1024/StackCube-v1/motionplanning/StackCube.h5 \
-c pd_ee_delta_pose \
-o rgb+depth+segmentation \
--no-vis \
--save-traj \
--verbose \
--no-allow-failure \
--num-envs 10

转化为zarr格式 
CUDA_VISIBLE_DEVICES=0 python scripts/gen_data/maniskill/h52zarr.py \
--input-path /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/MapPolicy/Data/maniskill_1024/StackCube-v1/motionplanning/StackCube.rgb+depth+segmentation.pd_ee_delta_pose.physx_cpu.h5 \
--zarr-save-dir /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/MapPolicy/data/maniskill/StackCube-v1 \
--max-episode 1000 \
--save-image-size 1024 \
--image-size 256 \
--num-points 4096 \
--num-workers 10


一键生成

CUDA_VISIBLE_DEVICES=0 python scripts/gen_data/maniskill/make_maniskill_zarr.py \
--task-name StackCube-v1 \
--num-traj 1000 \
--control-mode pd_ee_delta_pose \
--obs-mode rgb+depth+segmentation \
--traj-name StackCube \
--record-dir /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/MapPolicy/Data/maniskill_1024 \
--zarr-save-dir /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/MapPolicy/data/maniskill_1024