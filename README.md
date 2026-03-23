# 🚀 MapPolicy

## 🛠 1. Environment Setup

### 1.1 Conda
```bash
conda create --name mappolicy python=3.11 -y
conda activate mappolicy
```

### 1.2 初始化子模块
```bash
git submodule update --init --recursive
```

---

## 📦 2. 核心依赖安装

### 2.1 安装 PyTorch (以 CUDA 11.8 为例)
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

### 2.2 安装 3D 与几何处理库
```bash
# PyTorch3D
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation

# Open3D
pip install open3d

# PyTorch Geometric (PyG)
pip install torch_geometric --no-build-isolation
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html --no-build-isolation
```

### 2.3 Third Parties

FastSAM
```bash
cd mappolicy/models/fast_sam/FastSAM
pip install -r requirements.txt
cd ../../../..
```

---

## 🎮 3. 仿真环境配置 (Simulation)

### 3.1 MetaWorld
```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld

# ssh
pip install git+ssh://git@github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
```

### 3.2 RLBench

```bash
export COPPELIASIM_ROOT="your_path_to_coppeliasim"
# 下载并安装 CoppeliaSim
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT 
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# 安装 RLBench
cd third_party/RLBench
pip install -e .
cd ../..
```

### 3.3 ManiSkill
```bash
pip install --upgrade mani_skill

# use numpy < 2 if something wrong
pip install numpy==1.26.4
```

---

## 📊 4. Environment Variables

wandb

```bash
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_USER_EMAIL="your_wandb_email"
export WANDB_USERNAME="your_wandb_username"
```

root

```bash
export PYTHONPATH="your_path_to_MapPolicy"
export MAPPOLICY_ROOT="your_path_to_MapPolicy"
```

---

## 🚀 5. 启动训练 (Training)

项目目前支持三类训练范式：

### 5.1 MLP 作为 Policy Head

对应脚本：`mappolicy/train.py`（通过 `agent` 配置切换具体模型）

```bash
cd mappolicy

# 基础 MLP Policy Head（推荐显式指定）
CUDA_VISIBLE_DEVICES=0 python train.py agent=mappolicy_pcd/Map_MLP
```

---

### 5.2 Diffusion Policy 作为 Policy Head

对应脚本：`mappolicy/train_dp.py`

```bash
cd mappolicy

# Diffusion Policy（Hybrid Image）
CUDA_VISIBLE_DEVICES=0 python train_dp.py
```

---

### 5.3 RGBD → 点云 → PointNet/结构图（Structure Map）+ 约束损失训练

该范式会基于 RGBD 构建点云表示，结合 PointNet/图网络构建结构图，并通过 map/physical 等辅助损失进行联合训练。

可用两种方式：

```bash
cd mappolicy

# 端到端策略训练（含结构图约束）
CUDA_VISIBLE_DEVICES=0 python train.py agent=mappolicy_pcd/Map_GNN

# 仅训练结构图/点云重建分支
CUDA_VISIBLE_DEVICES=0 python train_map.py
```

> 说明：`train.py` 默认读取 `config/train_maniskill.yaml`；`train_dp.py` 默认读取 `config/traindp_maniskill.yaml`；`train_map.py` 默认读取 `config/train_map.yaml`。

---

### 💡 Tips
* You may do `export PYTHONPATH=MapPolicy_root_path`
* For PointNet testing, you may need:
    `export PYTHONPATH=$PYTHONPATH:MapPolicy_root_path/mappolicy/models/pointnet/Pointnet_Pointnet2_pytorch_models`

---

## 🧭 6. 代码结构速览（便于后续改进）

建议优先关注以下目录：

- `mappolicy/train.py`：常规策略训练入口（MLP/GNN/结构图辅助 loss）
- `mappolicy/train_dp.py`：Diffusion Policy 训练入口（含 EMA）
- `mappolicy/train_map.py`：结构图/点云重建分支训练入口
- `mappolicy/config/`：Hydra 配置中心（训练、agent、benchmark）
- `mappolicy/models/`：策略模型与编码器实现
- `mappolicy/loss/`：主任务 loss 与 map/physical 辅助约束
- `mappolicy/envs/`：评估器与环境封装
- `mappolicy/dataset/`：数据读取、切分与预处理

---

## ⚙️ 7. 配置系统（Hydra）使用建议

### 7.1 常用配置文件
- `config/train_maniskill.yaml`：常规训练默认配置
- `config/traindp_maniskill.yaml`：Diffusion Policy 默认配置
- `config/train_map.yaml`：结构图分支默认配置

### 7.2 推荐覆盖方式（避免硬编码）
```bash
# 切任务 / 相机 / 数据路径
python train.py task_name=PickCube-v1 camera_name=base_camera dataset_dir=/path/to/data.zarr

# 切 agent
python train.py agent=mappolicy_pcd/Map_MLP
python train.py agent=mappolicy_pcd/Map_GNN

# 切设备
python train.py device=cuda:0
```

---

## 🛠 8. 后续改进建议工作流

1. **先改配置，再改代码**：优先通过 `config/*.yaml` 完成实验切换。  
2. **保持维度自动推断**：state/action 维度优先从 dataloader sample 自动获取。  
3. **统一日志出口**：训练日志走 WandB + `Logger`，避免核心逻辑里直接 `print`。  
4. **Diffusion 分支保留 EMA**：`config.train.use_ema=True` 时保持 EMA 同步。  
5. **新增 loss 时遵循现有接口**：兼容 `preds` 为 Tensor / Tuple / Dict 三种输入格式。  

---

## ✅ 9. 最小自检清单

每次改动后建议至少检查：

- 能否正常启动训练（无 import/config 错误）
- train/val loss 是否正常下降或稳定
- WandB 指标是否完整上报
- 模型 checkpoint 是否按预期保存
- 评估脚本是否可复现实验结果

### for runing (zehao)
```bash
export MAPPOLICY_ROOT=your_path_to_project_root
export PYTHONPATH=$MAPPOLICY_ROOT
conda activate mappolicy
cd $MAPPOLICY_ROOT
```

---

## 🔁 Path guide（具体文件）

现在优先使用单一环境变量：`MAPPOLICY_ROOT`。  
大部分路径会从它自动拼接，不再需要逐个文件手改。

### 1) 必配（推荐）

- 环境变量：`MAPPOLICY_ROOT`
    - 用法见上方运行命令。

### 2) 会读取 `MAPPOLICY_ROOT` 的配置文件

- [mappolicy/config/train_map.yaml](mappolicy/config/train_map.yaml)
    - `project_root: ${oc.env:MAPPOLICY_ROOT,...}`
- [mappolicy/config/train_maniskill.yaml](mappolicy/config/train_maniskill.yaml)
    - `project_root: ${oc.env:MAPPOLICY_ROOT,...}`
- [mappolicy/config/traindp_maniskill.yaml](mappolicy/config/traindp_maniskill.yaml)
    - `project_root: ${oc.env:MAPPOLICY_ROOT,...}`
- [mappolicy/config/train_metaworld.yaml](mappolicy/config/train_metaworld.yaml)
    - `project_root: ${oc.env:MAPPOLICY_ROOT,...}`
- [mappolicy/config/train_rlbench.yaml](mappolicy/config/train_rlbench.yaml)
    - `project_root: ${oc.env:MAPPOLICY_ROOT,...}`

### 3) 会读取 `MAPPOLICY_ROOT` 的脚本/测试入口

- [mappolicy/models/mappolicy/map_constructor.py](mappolicy/models/mappolicy/map_constructor.py)
    - `--config_path` 默认值、FastSAM ckpt、test_output
- [mappolicy/models/fast_sam/fastsam_loader.py](mappolicy/models/fast_sam/fastsam_loader.py)
    - demo `main()` 的 ckpt/image/output
- [mappolicy/dataset/maniskill_dataset.py](mappolicy/dataset/maniskill_dataset.py)
    - `__main__` 测试数据路径
- [mappolicy/dataset/rlbench_dataset.py](mappolicy/dataset/rlbench_dataset.py)
    - `__main__` 测试数据路径
- [mappolicy/models/diffusion_policy/rgb_lowdim_encoder.py](mappolicy/models/diffusion_policy/rgb_lowdim_encoder.py)
    - `__main__` 配置路径
- [mappolicy/models/diffusion_policy/diffusion_unet_hybrid_image_policy.py](mappolicy/models/diffusion_policy/diffusion_unet_hybrid_image_policy.py)
    - `__main__` 配置/数据路径
- [mappolicy/envs/metaworld_env.py](mappolicy/envs/metaworld_env.py)
    - debug `.ply` 输出路径
- [scripts/gen_data/maniskill/make_maniskill_zarr.py](scripts/gen_data/maniskill/make_maniskill_zarr.py)
    - `--zarr-save-dir` 默认值
- [scripts/gen_data/maniskill/read_h5.py](scripts/gen_data/maniskill/read_h5.py)
    - 默认 `.h5` 路径
- [scripts/gen_data/maniskill/read_camera_params.py](scripts/gen_data/maniskill/read_camera_params.py)
    - 默认 `.h5` 路径
- [scripts/gen_data/rlbench/gen_data_rlbench.py](scripts/gen_data/rlbench/gen_data_rlbench.py)
    - `--rlbench-data-root` 默认值
- [scripts/gen_data/rlbench/gen_data_rlbench_keyframe.py](scripts/gen_data/rlbench/gen_data_rlbench_keyframe.py)
    - `--rlbench-data-root` 默认值

### 4) 仍可手动覆盖

如果你的数据不在 `$MAPPOLICY_ROOT` 下，可直接传命令行参数覆盖，例如：

- `dataset_dir=...`
- `--rlbench-data-root ...`
- `--zarr-save-dir ...`
- `--config_path ...`