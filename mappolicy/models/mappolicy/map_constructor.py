# Construct Structure Map
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import pathlib
import sys
models_path = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(
    str(models_path / "maps")
)

# MetaWorld
from mappolicy.maps.Basketball import StructureMap_Basketball
from mappolicy.maps.BinPicking import StructureMap_BinPicking
from mappolicy.maps.BoxClose import StructureMap_BoxClose
from mappolicy.maps.CoffeePull import StructureMap_CoffeePull
from mappolicy.maps.CoffeePush import StructureMap_CoffeePush
from mappolicy.maps.Disassemble import StructureMap_Disassemble
from mappolicy.maps.Hammer import StructureMap_Hammer
from mappolicy.maps.HandInsert import StructureMap_HandInsert
from mappolicy.maps.HandlePull import StructureMap_HandlePull
from mappolicy.maps.HandlePullSide import StructureMap_HandlePullSide
from mappolicy.maps.LeverPull import StructureMap_LeverPull
from mappolicy.maps.PegInsertSide import StructureMap_PegInsertSide
from mappolicy.maps.PegUnplugSide import StructureMap_PegUnplugSide
from mappolicy.maps.PickOutOfHole import StructureMap_PickOutOfHole
from mappolicy.maps.PickPlace import StructureMap_PickPlace
from mappolicy.maps.PickPlaceWall import StructureMap_PickPlaceWall
from mappolicy.maps.Push import StructureMap_Push
from mappolicy.maps.PushBack import StructureMap_PushBack
from mappolicy.maps.PushWall import StructureMap_PushWall
from mappolicy.maps.ReachWall import StructureMap_ReachWall
from mappolicy.maps.ShelfPlace import StructureMap_ShelfPlace
from mappolicy.maps.Soccer import StructureMap_Soccer
from mappolicy.maps.StickPull import StructureMap_StickPull
from mappolicy.maps.Sweep import StructureMap_Sweep
from mappolicy.maps.SweepInto import StructureMap_SweepInto

# RLBench
from mappolicy.maps.RLBench_CloseBox import StructureMap_CloseBox
from mappolicy.maps.RLBench_LaptopLid import StructureMap_LaptopLid
from mappolicy.maps.RLBench_PutRubbishIn import StructureMap_PutRubbishIn
from mappolicy.maps.RLBench_ToiletSeatDown import StructureMap_ToiletSeatDown
from mappolicy.maps.RLBench_UnplugCharger import StructureMap_UnplugCharger
from mappolicy.maps.RLBench_WaterPlants import StructureMap_WaterPlants

# ManiSkill
from mappolicy.maps.ManiSkill_PickCube import StructureMap_PickCube
from mappolicy.maps.ManiSkill_StackCube import StructureMap_StackCube
from mappolicy.maps.ManiSkill_PlugCharger import StructureMap_PlugCharger
from mappolicy.maps.ManiSkill_StackPyramid import StructureMap_StackPyramid
from mappolicy.maps.ManiSkill_PullCubeTool import StructureMap_PullCubeTool


##### clip
from mappolicy.models.Clip.clip_encoder import CLIPEncoder
# fast_sam
from mappolicy.models.fast_sam.fastsam_loader import FastSAM_Loader
# camera
from mappolicy.helper.graphics import get_pointcloud_from_input

MAP_DIM_VOCAB = {
    # metaworld
    "basketball": [8, 20, 44],
    "bin-picking": [19, 34, 64],
    "box-close": [20, 38, 74],
    "coffee-pull": [11, 23, 47],
    "coffee-push": [11, 23, 47],
    "disassemble": [6, 15, 33],
    "hammer": [6, 15, 33],
    "hand-insert": [6, 12, 24],
    "handle-pull": [6, 15, 33],
    "handle-pull-side": [6, 15, 33],
    "lever-pull": [8, 20, 44],
    "peg-insert-side": [8, 17, 35],
    "peg-unplug-side": [7, 16, 34],
    "pick-out-of-hole": [9, 18, 36],
    "pick-place": [3, 6, 12],
    "pick-place-wall": [9, 18, 36],
    "push": [3, 6, 12],
    "push-back": [3, 6, 12],
    "push-wall": [9, 18, 36],
    "reach-wall": [6, 12, 24],
    "shelf-place": [14, 26, 50],
    "soccer": [13, 28, 58],
    "stick-pull": [11, 23, 47],
    "sweep": [3, 6, 12],
    "sweep-into": [9, 18, 36],
    
    # rlbench
    "close_box": [11, 20, 38],
    "close_laptop_lid": [9, 18, 36],
    "put_rubbish_in_bin": [11, 23, 47],
    "toilet_seat_down": [10, 22, 46],
    "unplug_charger": [7, 16, 34],
    "water_plants": [13, 28, 58],
    
    # Maniskill
    "PickCube-v1": [3, 6, 12],
    "PegInsertionSide-v1": [8, 17, 35],
    "StackCube-v1": [9, 18, 36],
    "PlugCharger-v1": [18, 36, 72],
    "StackPyramid-v1": [9, 18, 36],
    "PullCubeTool-v1": [7, 16, 34],
}
MAP_CLASS_VOCAB = {
    # metaworld
    "basketball": StructureMap_Basketball,
    "bin-picking": StructureMap_BinPicking,
    "box-close": StructureMap_BoxClose,
    "coffee-pull": StructureMap_CoffeePull,
    "coffee-push": StructureMap_CoffeePush,
    "disassemble": StructureMap_Disassemble,
    "hammer": StructureMap_Hammer,
    "hand-insert": StructureMap_HandInsert,
    "handle-pull": StructureMap_HandlePull,
    "handle-pull-side": StructureMap_HandlePullSide,
    "lever-pull": StructureMap_LeverPull,
    "peg-insert-side": StructureMap_PegInsertSide,
    "peg-unplug-side": StructureMap_PegUnplugSide,
    "pick-out-of-hole": StructureMap_PickOutOfHole,
    "pick-place": StructureMap_PickPlace,
    "pick-place-wall": StructureMap_PickPlaceWall,
    "push": StructureMap_Push,
    "push-back": StructureMap_PushBack,
    "push-wall": StructureMap_PushWall,
    "reach-wall": StructureMap_ReachWall,
    "shelf-place": StructureMap_ShelfPlace,
    "soccer": StructureMap_Soccer,
    "stick-pull": StructureMap_StickPull,
    "sweep": StructureMap_Sweep,
    "sweep-into": StructureMap_SweepInto,
    
    # rlbench
    "close_box": StructureMap_CloseBox,
    "close_laptop_lid": StructureMap_LaptopLid,
    "put_rubbish_in_bin": StructureMap_PutRubbishIn,
    "toilet_seat_down": StructureMap_ToiletSeatDown,
    "unplug_charger": StructureMap_UnplugCharger,
    "water_plants": StructureMap_WaterPlants,

    # Maniskill
    "PickCube-v1": StructureMap_PickCube,
    "PegInsertionSide-v1": StructureMap_PegInsertSide,
    "StackCube-v1": StructureMap_StackCube,
    "PlugCharger-v1": StructureMap_PlugCharger,
    "StackPyramid-v1": StructureMap_StackPyramid,
    "PullCubeTool-v1": StructureMap_PullCubeTool
}

class ParameterEstimator_SingleFrame(nn.Module):
    def __init__ (self,
                  point_cloud_encoder: nn.Module,
                  task_name,
                  device,
                  ):
        if task_name not in MAP_DIM_VOCAB or task_name not in MAP_CLASS_VOCAB:
            raise ValueError(f"Unknown task_name: {task_name}. Available: {list(MAP_CLASS_VOCAB.keys())}")
        self.map_name = task_name
        self.dims = MAP_DIM_VOCAB[task_name]
        self.device = device
        self.MapClass = MAP_CLASS_VOCAB[task_name]
        super(ParameterEstimator_SingleFrame, self).__init__()
        self.clip_encoder = CLIPEncoder("ViT-B/32").to(self.device)
        self.point_cloud_encoder = point_cloud_encoder
        self.estimation_head = nn.Sequential(
            nn.Linear(point_cloud_encoder.feature_dim, self.dims[2]),
        )
        
    def forward(self, point_cloud):
        features = self.point_cloud_encoder(point_cloud)
        parameters = self.estimation_head(features)
        sizes = parameters[:, 0:self.dims[0]]
        positions = parameters[:, self.dims[0]:self.dims[1]]
        rotations = parameters[:, self.dims[1]:self.dims[2]]
        scene_map = self.MapClass(sizes, positions, rotations, self.clip_encoder, preprocess=False)
        return scene_map
    
class ParameterEstimator_SingleFrame_Regularization(nn.Module):
    def __init__ (self,
                  point_cloud_encoder: nn.Module,
                  task_name: str,
                  device,
                  ):
        if task_name not in MAP_DIM_VOCAB or task_name not in MAP_CLASS_VOCAB:
            raise ValueError(f"Unknown task_name: {task_name}. Available: {list(MAP_CLASS_VOCAB.keys())}")
        self.map_name = task_name
        self.dims = MAP_DIM_VOCAB[task_name]
        self.device = device
        self.MapClass = MAP_CLASS_VOCAB[task_name]
        super().__init__()
        self.clip_encoder = CLIPEncoder("ViT-B/32").to(self.device)
        self.point_cloud_encoder = point_cloud_encoder
        self.estimation_head = nn.Sequential(
            nn.Linear(point_cloud_encoder.feature_dim, self.dims[2]),
        )
        
    def forward(self, point_cloud):
        features = self.point_cloud_encoder(point_cloud)
        parameters = self.estimation_head(features)
        sizes = parameters[:, 0:self.dims[0]]
        positions = parameters[:, self.dims[0]:self.dims[1]]
        rotations = parameters[:, self.dims[1]:self.dims[2]]
        scene_map = self.MapClass(sizes, positions, rotations, self.clip_encoder, preprocess=True)
        return scene_map
    
class ParameterEstimator_SingleFrame_Segmentation(nn.Module):
    def __init__ (self,
                  point_cloud_encoder: nn.Module,
                  benchmark_name: str,
                  camera_name: str,
                  task_name: str,
                  fastsam_ckpt_path: str,
                  image_size: int,
                  device,
                  ):
        self.benchmark_name = benchmark_name
        self.camera_name = camera_name
        if task_name not in MAP_DIM_VOCAB or task_name not in MAP_CLASS_VOCAB:
            raise ValueError(f"Unknown task_name: {task_name}. Available: {list(MAP_CLASS_VOCAB.keys())}")
        self.map_name = task_name
        self.dims = MAP_DIM_VOCAB[task_name]
        self.device = device
        self.MapClass = MAP_CLASS_VOCAB[task_name]
        super().__init__()
        self.clip_encoder = CLIPEncoder("ViT-B/32").to(self.device)
        self.segmentation_model = FastSAM_Loader(ckpt_path=fastsam_ckpt_path, device=device, imgsz=image_size)
        self.point_cloud_encoder = point_cloud_encoder
        self.estimation_head = nn.Sequential(
            nn.Linear(point_cloud_encoder.feature_dim, self.dims[2]),
        )
        self.num_points = 1024
        self.subgraph_dict = self._build_subgraph_dict()

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _build_subgraph_dict(self):
        """
        从 map template 中提取子图语义，并组织为 dict。

        优先级：
        1) map 显式提供的粗粒度子图定义（推荐）
        2) 回退到按 Node_Semantic 聚合（细粒度）

        目标格式:
        {
            "subgraph_0": {"text_prompt": "red cube", "node_indices": [0]},
            ...
        }
        """
        size_dim = self.dims[0]
        pos_dim = self.dims[1] - self.dims[0]
        rot_dim = self.dims[2] - self.dims[1]

        sizes = torch.ones((1, size_dim), dtype=torch.float32, device=self.device) * 0.1
        positions = torch.zeros((1, pos_dim), dtype=torch.float32, device=self.device)
        rotations = torch.zeros((1, rot_dim), dtype=torch.float32, device=self.device)

        # 使用 6D rotation 的单位表示 [1,0,0,0,1,0]
        if rot_dim % 6 == 0:
            for i in range(rot_dim // 6):
                rotations[:, i * 6 : (i + 1) * 6] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    dtype=torch.float32,
                    device=self.device,
                )

        template_map = self.MapClass(sizes, positions, rotations, self.clip_encoder, preprocess=True)

        # 1) 优先使用 map 内定义的粗粒度子图
        # 约定字段：template_map.Subgraph_Prompts = [{"text_prompt": str, "node_indices": List[int]}, ...]
        if hasattr(template_map, "Subgraph_Prompts") and template_map.Subgraph_Prompts is not None:
            subgraph_dict = OrderedDict()
            for i, item in enumerate(template_map.Subgraph_Prompts):
                subgraph_dict[f"subgraph_{i}"] = {
                    "text_prompt": item["text_prompt"],
                    "node_indices": item["node_indices"],
                }
            return subgraph_dict

        # 2) 回退：按节点语义聚合（细粒度）
        sem_to_indices = OrderedDict()
        for idx, node in enumerate(template_map.Node):
            sem = node.Node_Semantic
            if sem not in sem_to_indices:
                sem_to_indices[sem] = []
            sem_to_indices[sem].append(idx)

        subgraph_dict = OrderedDict()
        for i, (sem, indices) in enumerate(sem_to_indices.items()):
            subgraph_dict[f"subgraph_{i}"] = {
                "text_prompt": sem,
                "node_indices": indices,
            }
        return subgraph_dict

    def _prepare_point_cloud_for_encoder(self, points_xyz: np.ndarray):
        """
        将投影得到的点云整理为 PointNet 输入 [1, N, 6]。
        前 3 维是 xyz，后 3 维填充 0。
        """
        if points_xyz is None or points_xyz.shape[0] == 0:
            points_xyz = np.zeros((1, 3), dtype=np.float32)

        points_xyz = np.asarray(points_xyz, dtype=np.float32)
        n = points_xyz.shape[0]

        if n >= self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
        else:
            idx = np.random.choice(n, self.num_points, replace=True)
        points_xyz = points_xyz[idx]

        points = np.concatenate(
            [points_xyz, np.zeros((self.num_points, 3), dtype=np.float32)], axis=1
        )
        points = torch.from_numpy(points).unsqueeze(0).to(self.device)
        return points
        
    def forward(self, rgb, depth):
        rgb_np = self._to_numpy(rgb)
        depth_np = self._to_numpy(depth)

        if rgb_np.ndim == 4:
            rgb_np = rgb_np[0]
        if depth_np.ndim == 3:
            depth_np = depth_np[0]

        masks_dict = OrderedDict()
        partial_point_cloud_dict = OrderedDict()
        merged_partial_point_clouds = []

        subgraph_items = list(self.subgraph_dict.items())
        text_prompts = [item[1]["text_prompt"] for item in subgraph_items]

        prompt_to_mask = self.segmentation_model.segment_multi_text_prompts(
            rgb_np,
            text_prompts,
            topk=8,
        )

        # 1) 从 map 获取子图 dict；2) 每个子图按 text prompt 分割；3) projection 成 partial point cloud
        for subgraph_name, subgraph_info in subgraph_items:
            prompt = subgraph_info["text_prompt"]
            mask = prompt_to_mask.get(prompt, np.zeros(depth_np.shape, dtype=bool))
            masks_dict[subgraph_name] = mask

            partial_pc = get_pointcloud_from_input(
                rgb_np,
                depth_np,
                mask,
                self.benchmark_name,
                self.camera_name,
            )
            partial_point_cloud_dict[subgraph_name] = partial_pc

            if partial_pc is not None and partial_pc.shape[0] > 0:
                merged_partial_point_clouds.append(partial_pc)

        if len(merged_partial_point_clouds) > 0:
            merged_pc = np.concatenate(merged_partial_point_clouds, axis=0)
        else:
            merged_pc = np.zeros((1, 3), dtype=np.float32)

        point_cloud = self._prepare_point_cloud_for_encoder(merged_pc)

        # 4) 用 pointnet 编码并回归参数
        features = self.point_cloud_encoder(point_cloud)
        parameters = self.estimation_head(features)
        sizes = parameters[:, 0:self.dims[0]]
        positions = parameters[:, self.dims[0]:self.dims[1]]
        rotations = parameters[:, self.dims[1]:self.dims[2]]

        # 便于调试可视化
        self.latest_masks_dict = masks_dict
        self.latest_partial_point_cloud_dict = partial_point_cloud_dict

        scene_map = self.MapClass(sizes, positions, rotations, self.clip_encoder, preprocess=True)
        return scene_map


def test_parameter_estimator_singleframe_segmentation(
    config_path: str,
    split: str = "train",
    fastsam_ckpt_path: str = None,
    device: str = None,
    run_forward: bool = True,
    output_dir: str = None,
    num_cases: int = 5,
):
    """
    使用 train_map.yaml 风格配置加载 dataset，并测试
    ParameterEstimator_SingleFrame_Segmentation。
    """
    import os
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from mappolicy.models.pointnet.model_loader import PointnetEnc

    # 使用 Hydra 组合配置，确保 defaults (agent/benchmark 等) 被正确展开
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)

    if device is None:
        device = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")

    if fastsam_ckpt_path is None:
        project_root = cfg.get("project_root", os.getenv("MAPPOLICY_ROOT", "your_path_to_project_root"))
        fastsam_ckpt_path = f"{project_root}/mappolicy/models/fast_sam/FastSAM/weights/FastSAM-x.pt"

    if output_dir is None:
        project_root = cfg.get("project_root", os.getenv("MAPPOLICY_ROOT", "your_path_to_project_root"))
        output_dir = f"{project_root}/test_output"

    dataset = instantiate(
        config=cfg.benchmark.dataset_instantiate_config,
        data_dir=cfg.dataset_dir,
        split=split,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    task_name = cfg.task_name
    benchmark_name = cfg.benchmark.name
    camera_name = cfg.camera_name
    image_size = int(cfg.image_size)

    estimator = ParameterEstimator_SingleFrame_Segmentation(
        point_cloud_encoder=PointnetEnc().to(device),
        benchmark_name=benchmark_name,
        camera_name=camera_name,
        task_name=task_name,
        fastsam_ckpt_path=fastsam_ckpt_path,
        image_size=image_size,
        device=device,
    ).to(device)
    estimator.eval()

    print("[SegEstimatorTest] build success")
    print(f"[SegEstimatorTest] task={task_name}, benchmark={benchmark_name}, camera={camera_name}")
    print(f"[SegEstimatorTest] subgraphs={len(estimator.subgraph_dict)}")
    for k, v in estimator.subgraph_dict.items():
        print(f"  - {k}: prompt='{v['text_prompt']}', nodes={v['node_indices']}")

    if not run_forward:
        print("[SegEstimatorTest] skip forward (--run_forward false)")
        return estimator

    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    scene_map = None
    saved_cases = 0

    for case_idx, batch in enumerate(loader):
        if saved_cases >= int(num_cases):
            break

        # 兼容当前数据集定义
        # ManiSkill: image, depth, point_cloud, point_cloud_no_robot, robot_state, action
        # 其余: image, point_cloud, point_cloud_no_robot, robot_state, ..., action, text
        if not (isinstance(batch, (tuple, list)) and len(batch) >= 2):
            continue

        image = batch[0]
        if len(batch) == 6:
            depth = batch[1]
        else:
            depth = None

        if depth is None:
            continue

        rgb = image[0].permute(1, 2, 0).detach().cpu().numpy()
        if np.issubdtype(rgb.dtype, np.floating):
            if rgb.max() <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0.0, 255.0)
        rgb = rgb.astype(np.uint8)
        depth_np = depth[0].detach().cpu().numpy()

        with torch.no_grad():
            scene_map = estimator(rgb, depth_np)

        case_dir = os.path.join(output_dir, f"case_{case_idx:04d}")
        os.makedirs(case_dir, exist_ok=True)

        rgb_u8 = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
        Image.fromarray(rgb_u8).save(os.path.join(case_dir, "original_rgb.png"))

        for subgraph_name, mask in estimator.latest_masks_dict.items():
            mask_bool = np.asarray(mask).astype(bool)
            masked_rgb = rgb_u8.copy()
            masked_rgb[~mask_bool] = 0

            mask_img_path = os.path.join(case_dir, f"{subgraph_name}_mask.png")
            masked_rgb_path = os.path.join(case_dir, f"{subgraph_name}_masked_rgb.png")
            partial_pc_path = os.path.join(case_dir, f"{subgraph_name}_partial_pc.npy")

            Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(mask_img_path)
            Image.fromarray(masked_rgb).save(masked_rgb_path)

            partial_pc = estimator.latest_partial_point_cloud_dict.get(subgraph_name, None)
            if partial_pc is None:
                partial_pc = np.zeros((0, 3), dtype=np.float32)
            np.save(partial_pc_path, np.asarray(partial_pc, dtype=np.float32))

        print(
            f"[SegEstimatorTest] case={case_idx} forward success: "
            f"N_nodes={scene_map.N}, N_edges={scene_map.M}, "
            f"saved={case_dir}"
        )
        saved_cases += 1

    if scene_map is None:
        print("[SegEstimatorTest] no valid case saved (depth may be unavailable)")
        return estimator

    print(f"[SegEstimatorTest] masks={list(estimator.latest_masks_dict.keys())}")
    print(f"[SegEstimatorTest] saved {saved_cases} case(s) to: {output_dir}")
    return scene_map


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Test ParameterEstimator_SingleFrame_Segmentation with dataset from config"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=f"{os.getenv('MAPPOLICY_ROOT', 'your_path_to_project_root')}/mappolicy/config/train_map.yaml",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--fastsam_ckpt_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_forward", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--output_dir", type=str, default=f"{os.getenv('MAPPOLICY_ROOT', 'your_path_to_project_root')}/mappolicy/models/mappolicy/test_output")
    parser.add_argument("--num_cases", type=int, default=5)
    args = parser.parse_args()

    test_parameter_estimator_singleframe_segmentation(
        config_path=args.config_path,
        split=args.split,
        fastsam_ckpt_path=args.fastsam_ckpt_path,
        device=args.device,
        run_forward=args.run_forward,
        output_dir=args.output_dir,
        num_cases=args.num_cases,
    )