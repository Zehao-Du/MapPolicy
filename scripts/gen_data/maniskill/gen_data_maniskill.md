生成原始.h5数据

```bash
CUDA_VISIBLE_DEVICES=9 python -m mani_skill.examples.motionplanning.panda.run --env-id "StackCube-v1" --num-traj 1100 --only-count-success --save-video --record-dir Data/maniskill --traj-name StackCube --num-procs 10
```

replay数据并获取点云，更改控制方式

```bash
CUDA_VISIBLE_DEVICES=9 python -m mani_skill.trajectory.replay_trajectory \
  --traj-path /data2/zehao/MapPolicy/Data/maniskill/StackCube-v1/motionplanning/StackCube.h5 \
  -c pd_ee_delta_pose \
  -o rgb+depth+segmentation \
  --no-vis \
  --save-traj \
  --verbose \
  --no-allow-failure \
  --num-envs 10
```

转化为zarr格式
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/gen_data/maniskill/h52zarr.py \
--input-path /data2/zehao/MapPolicy/Data/maniskill/PickCube-v1/motionplanning/PickCube.rgb+depth+segmentation.pd_ee_delta_pose.physx_cpu.h5 \
--zarr-save-dir /data2/zehao/MapPolicy/data/maniskill/PickCube-v1 \
--max-episode 1000
```

Change image size at 
```/data2/zehao/miniconda3/envs/mappolicy/lib/python3.11/site-packages/mani_skill/envs/tasks/tabletop```


h5：
```
traj keys ['obs', 'actions', 'terminated', 'truncated', 'success', 'env_states']
obs keys ['agent', 'extra', 'sensor_param', 'sensor_data']
sensor_data keys ['base_camera']
cam keys ['segmentation', 'rgb', 'depth']
seg shape (75, 224, 224, 1) int32 min/max 1 18
unique first10 len 12 sample [ 1  2  3  6  7  8 10 12 14 16 17 18]
top ids step0 [(16, 32705), (17, 12074), (6, 1861), (7, 1513), (10, 566), (2, 450), (3, 449), (8, 423), (18, 75), (12, 25), (14, 22), (1, 13)]
extra keys ['is_grasped', 'tcp_pose', 'goal_pos']

sensor_param keys ['extrinsic_cv', 'cam2world_gl', 'intrinsic_cv']
extrinsic_cv (75, 3, 4) float32
cam2world_gl (75, 4, 4) float32
intrinsic_cv (75, 3, 3) float32
```