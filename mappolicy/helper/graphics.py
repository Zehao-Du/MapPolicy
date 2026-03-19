import numpy as np
import open3d as o3d
import pytorch3d.ops as torch3d_ops
import torch
from scipy.spatial.transform import Rotation


class BasePose(object):
    @staticmethod
    def pose_delta_2d(pose1_2d, pose2_2d):
        delta_pose = pose2_2d - pose1_2d
        return delta_pose


class HomogeneousCoordinates(object):
    @staticmethod
    def pose_7DoF_to_matrix(pose_7d, scalar_first=False):
        """Convert 7DoF pose to homogeneous matrix.

        Args:
            pose_7d (np.array): [x, y, z, qx, qy, qz, qw] if scalar_first is False
            scalar_first (bool, optional): Whether the scalar component goes first or last. Defaults to False.

        Returns:
            np.array: 4x4 homogeneous matrix
        """
        pos, quat = pose_7d[:3], pose_7d[3:]
        rotation_matrix = Rotation.from_quat(
            quat, scalar_first=scalar_first
        ).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = pos
        return matrix

    @staticmethod
    def maxtrix_to_pose_7DoF(matrix, scalar_first=False):
        """Convert homogeneous matrix to 7DoF pose.

        Args:
            matrix (np.array): 4x4 homogeneous matrix
            scalar_first (bool, optional): Whether the scalar component goes first or last. Defaults to False.

        Returns:
            np.array: [x, y, z, qx, qy, qz, qw] if scalar_first is False
        """
        pos = matrix[:3, 3]
        quat = Rotation.from_matrix(matrix[:3, :3]).as_quat(scalar_first=scalar_first)
        pose = np.concatenate([pos, quat])
        return pose

    @staticmethod
    def pos_rot_to_matrix(pos, rot):
        """Convert position and rotation to homogeneous matrix.

        Args:
            pos (np.array): [x, y, z]
            rot (np.array): 3x3 rotation matrix

        Returns:
            np.array: 4x4 homogeneous matrix
        """
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rot
        homogeneous_matrix[:3, 3] = pos
        return homogeneous_matrix


class EEpose(object):
    @staticmethod
    def pose_delta_7DoF(pose1_7d, pose2_7d, scalar_first=False):
        """Calculate the relative pose between two poses.

        Args:
            pose1_7d (np.array): [x, y, z, qx, qy, qz, qw] if scalar_first is False
            pose2_7d (np.array): [x, y, z, qx, qy, qz, qw] if scalar_first is False
            scalar_first (bool, optional): Whether the scalar component goes first or last. Defaults to False.

        Returns:
            np.array: [dx, dy, dz, dqx, dqy, dqz, dw] if scalar_first is False
        """
        position1 = pose1_7d[:3]
        position2 = pose2_7d[:3]
        delta_postion = position2 - position1
        quat1 = Rotation.from_quat(pose1_7d[3:], scalar_first=scalar_first)
        quat2 = Rotation.from_quat(pose2_7d[3:], scalar_first=scalar_first)
        delta_quat = (quat2 * (quat1.inv())).as_quat(scalar_first=scalar_first)
        delta_7DoF = np.concatenate([delta_postion, delta_quat])
        return delta_7DoF

    @staticmethod
    def pose_6DoF_to_7DoF(pose_6d, scalar_first=False, degrees=False):
        """Convert 6DoF pose to 7DoF pose.

        Args:
            pose_6d (np.array): [x, y, z, roll, pitch, yaw]
            scalar_first (bool, optional): Whether the scalar component goes first or last. Defaults to False.

        Returns:
            np.array: [x, y, z, qx, qy, qz, qw] if scalar_first is False
        """
        x, y, z, roll, pitch, yaw = pose_6d
        delta_rotation = Rotation.from_euler(
            "xyz", [roll, pitch, yaw], degrees=degrees
        ).as_quat(scalar_first=scalar_first)
        pose_7d = [
            x,
            y,
            z,
            delta_rotation[0],
            delta_rotation[1],
            delta_rotation[2],
            delta_rotation[3],
        ]
        return pose_7d

    @staticmethod
    def pose_7DoF_to_6DoF(pose_7d, scalar_first=False, degrees=False):
        """Convert 7DoF pose to 6DoF pose.

        Args:
            pose_7d (np.array): [x, y, z, qx, qy, qz, qw] if scalar_first is False
            scalar_first (bool, optional): Whether the scalar component goes first or last. Defaults to False.
            degrees (bool, optional): Whether the euler angles are in degrees. Defaults to False.

        Returns:
            np.array: [x, y, z, roll, pitch, yaw]
        """
        position = pose_7d[:3]
        rotation = Rotation.from_quat(pose_7d[3:], scalar_first=scalar_first).as_quat(
            scalar_first=scalar_first
        )
        euler_angles = Rotation.from_quat(rotation, scalar_first=scalar_first).as_euler(
            "xyz", degrees=degrees
        )
        pose_6DoF = np.concatenate([position, euler_angles])
        return pose_6DoF

    @staticmethod
    def pose_add_7DoF(pose1_7d, pose2_7d, scalar_first=False):
        """Add two poses.

        Args:
            pose1_7d (np.array): [x, y, z, qx, qy, qz, qw] if scalar_first is False
            pose2_7d (np.array): [x, y, z, qx, qy, qz, qw] if scalar_first is False
            scalar_first (bool, optional): Whether the scalar component goes first or last. Defaults to False.

        Returns:
            np.array: [x, y, z, qx, qy, qz, qw] if scalar_first is False
        """
        position1 = pose1_7d[:3]
        position2 = pose2_7d[:3]
        position = position1 + position2
        quat1 = Rotation.from_quat(pose1_7d[3:], scalar_first=scalar_first)
        quat2 = Rotation.from_quat(pose2_7d[3:], scalar_first=scalar_first)
        quat = quat1 * quat2
        pose_7d = np.concatenate([position, quat.as_quat(scalar_first=scalar_first)])
        return pose_7d

    @staticmethod
    def calculate_child_pose_after_ancestor_moving_7DoF(
        initial_pose_child, initial_pose_ancestor, new_pose_ancestor, scalar_first=False
    ):
        """Calculate the new pose of the child after the ancestor has moved.

        Args:
            initial_pose_child (np.array): [x, y, z, qx, qy, qz, qw] if scalar_first is False
            initial_pose_ancestor (np.array): [x, y, z, qx, qy, qz, qw] if scalar_first is False
            new_pose_ancestor (np.array): [x, y, z, qx, qy, qz, qw] if scalar_first is False
            scalar_first (bool, optional): Whether the scalar component goes first or last. Defaults to False.

        Returns:
            np.array: [x, y, z, qx, qy, qz, qw] if scalar_first is False
        """
        maxtrix_ancestor_initial = HomogeneousCoordinates.pose_7DoF_to_matrix(
            initial_pose_ancestor, scalar_first=scalar_first
        )
        maxtrix_ancestor_new = HomogeneousCoordinates.pose_7DoF_to_matrix(
            new_pose_ancestor, scalar_first=scalar_first
        )
        maxtrix_child_initial = HomogeneousCoordinates.pose_7DoF_to_matrix(
            initial_pose_child, scalar_first=scalar_first
        )
        maxtrix_child_new = (
            maxtrix_ancestor_new
            @ np.linalg.inv(maxtrix_ancestor_initial)
            @ maxtrix_child_initial
        )
        pose_child_new = HomogeneousCoordinates.maxtrix_to_pose_7DoF(
            maxtrix_child_new, scalar_first=scalar_first
        )
        return pose_child_new


class Quaternion(object):
    @staticmethod
    def normalize_quaternion(quaternion):
        """
        Normalize the quaternion to make it a unit quaternion.
        """
        q = np.array(quaternion)
        norm = np.linalg.norm(q)
        if norm == 0:
            return q
        return q / norm

    @staticmethod
    def ensure_positive_real_part(quaternion, scalar_first=False):
        """Ensure the real part of the quaternion is positive.

        Args:
            quaternion (np.array): [qx, qy, qz, qw] if scalar_first is False
            scalar_first (bool, optional): Whether the scalar component goes first or last. Defaults to False.

        Returns:
            np.array: [qx, qy, qz, qw] if scalar_first is False
        """
        real_part = quaternion[0] if scalar_first else quaternion[-1]
        if real_part < 0:
            quaternion = -quaternion
        return quaternion


class Camera(object):
    @staticmethod
    def camera_matrix_to_o3d(cam_mat, width, height):
        cx = cam_mat[0, 2]
        fx = cam_mat[0, 0]
        cy = cam_mat[1, 2]
        fy = cam_mat[1, 1]
        return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


class PointCloud(object):
    @staticmethod
    def o3d_to_numpy(o3d_point_cloud):
        points_np = np.asarray(o3d_point_cloud.points)
        colors_np = np.asarray(o3d_point_cloud.colors)
        point_cloud_np = np.hstack([points_np, colors_np])
        return point_cloud_np

    @staticmethod
    def point_cloud_sampling(
        point_cloud: np.ndarray, num_points: int, method: str = "fps"
    ):
        """
        support different point cloud sampling methods
        point_cloud: (N, 6), xyz+rgb or (N, 3), xyz
        """
        if num_points == "all":  # use all points
            return point_cloud

        if point_cloud.shape[0] <= num_points:
            # cprint(f"warning: point cloud has {point_cloud.shape[0]} points, but we want to sample {num_points} points", 'yellow')
            # pad with zeros
            point_cloud_dim = point_cloud.shape[-1]
            point_cloud = np.concatenate(
                [
                    point_cloud,
                    np.zeros((num_points - point_cloud.shape[0], point_cloud_dim)),
                ],
                axis=0,
            )
            return point_cloud

        if method == "uniform":
            # uniform sampling
            sampled_indices = np.random.choice(
                point_cloud.shape[0], num_points, replace=False
            )
            point_cloud = point_cloud[sampled_indices]
        elif method == "fps":
            # fast point cloud sampling using torch3d
            point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
            num_points = torch.tensor([num_points]).cuda()
            # remember to only use coord to sample
            _, sampled_indices = torch3d_ops.sample_farthest_points(
                points=point_cloud[..., :3], K=num_points
            )
            point_cloud = point_cloud.squeeze(0).cpu().numpy()
            point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
        else:
            raise NotImplementedError(
                f"point cloud sampling method {method} not implemented"
            )

        return point_cloud

    @staticmethod
    def normalize(pc):
        if pc.dim() == 3 and pc.shape[2] == 3:  # Shape is (b, n, 3)
            # Normalize coordinates directly
            coords = pc
            centroid = torch.mean(coords, dim=1, keepdim=True)  # (b, 1, 3)
            coords = coords - centroid  # Centering each point cloud
            m = torch.max(
                torch.sqrt(torch.sum(coords**2, dim=-1)), dim=1, keepdim=True
            )[
                0
            ]  # (b, 1)
            pc = coords / m.unsqueeze(-1)  # Normalize each point cloud

        elif pc.dim() == 3 and pc.shape[2] == 6:  # Shape is (b, n, 6)
            # Separate coordinates and features, normalize only coordinates
            coords = pc[:, :, :3]  # (b, n, 3) - coordinates
            features = pc[:, :, 3:]  # (b, n, 3) - other features
            centroid = torch.mean(coords, dim=1, keepdim=True)  # (b, 1, 3)
            coords = coords - centroid  # Centering each point cloud
            m = torch.max(
                torch.sqrt(torch.sum(coords**2, dim=-1)), dim=1, keepdim=True
            )[
                0
            ]  # (b, 1)
            coords = coords / m.unsqueeze(-1)  # Normalize each point cloud
            pc = torch.cat([coords, features], dim=-1)  # (b, n, 6)

        elif pc.dim() == 2 and pc.shape[1] == 3:  # Shape is (n, 3)
            # Normalize coordinates directly
            coords = pc
            centroid = torch.mean(coords, dim=0)  # (3,)
            coords = coords - centroid  # Centering the point cloud
            m = torch.max(torch.sqrt(torch.sum(coords**2, dim=1)))  # Scalar
            pc = coords / m  # Normalize the point cloud

        elif pc.dim() == 2 and pc.shape[1] == 6:  # Shape is (n, 6)
            # Separate coordinates and features, normalize only coordinates
            coords = pc[:, :3]  # (n, 3) - coordinates
            features = pc[:, 3:]  # (n, 3) - other features
            centroid = torch.mean(coords, dim=0)  # (3,)
            coords = coords - centroid  # Centering the point cloud
            m = torch.max(torch.sqrt(torch.sum(coords**2, dim=1)))  # Scalar
            coords = coords / m  # Normalize the point cloud
            pc = torch.cat([coords, features], dim=-1)  # (n, 6)

        else:
            raise ValueError(
                f"Input point cloud should have shape (n, 3) or (n, 6) or (b, n, 3) or (b, n, 6). But got {pc.shape}"
            )

        return pc

# Camera
# metaworld, camera extrinsic matrix
Metaworld_extrinsic_matrix = {
    "corner": np.array(
        [
            [-0.7071, 0.7071, 0.0, -0.495],
            [0.1925, 0.1925, 0.9623, -0.2887],
            [0.6804, 0.6804, -0.2722, 1.1839],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "corner2": np.array(
        [
            [-0.5499, -0.8332, 0.0584, 0.484],
            [-0.3762, 0.3095, 0.8733, -0.4096],
            [-0.7457, 0.4582, -0.4837, 1.5931],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}

# metaworld, camera intrinsic matrix
Metaworld_intrinsic_matrix = {
    "corner": np.array(
        [
            [270.3919, 0.0, 112.0],
            [0.0, 270.3919, 112.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "corner2": np.array(
        [
            [193.9897, 0.0, 112.0],
            [0.0, 193.9897, 112.0],
            [0.0, 0.0, 1.0],
        ]
    ),
}

# The coordinate frame in Sapien is: x(forward), y(left), z(upward)
# rlbench, camera extrinsic matrix
RLBench_extrinsic_matrix = {
    "front": np.array(
        [
            [ 1.16849501e-07, -1.00000000e+00, -6.03176614e-07,  8.32426571e-07,],
            [-4.22617918e-01, -5.63571533e-07,  9.06307948e-01, -8.61432102e-01,],
            [-9.06307948e-01,  1.31264604e-07, -4.22617948e-01,  1.89125107e+00,],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,],
        ]
    ),
}

# rlbench, camera intrinsic matrix
RLBench_intrinsic_matrix = {
    "front": np.array(
        [
            [-307.7174807,0.0,112.0,],
            [0.0,-307.7174807,112.0,],
            [0.0,0.0,1.0,],
        ]
    )
}

# The coordinate frame in ManiSkill2 is: x(forward), y(right), z(downward)
# maniskill, camera extrinsic matrix
ManiSkill_extrinsic_matrix = {
    "base_camera": np.array(
        [
            [ 0.0,        1.0,        0.0,          0.0      ],
            [ 0.78086877,  0.0,      -0.62469506,  0.14055645],
            [-0.62469506,  0.0,      -0.78086877,  0.6559298 ],
            [ 0.0,        0.0,        0.0,          1.0      ],
        ]
    )
}

# maniskill, camera intrinsic matrix
ManiSkill_intrinsic_matrix = {
    "base_camera": np.array(
        [
            [112.0,   0.0,  112.0],
            [  0.0, 112.0, 112.0],
            [  0.0,   0.0,   1.0]
        ]
    )
}


CAMERA_CONFIGS = {
    "metaworld": {
        "extrinsic": Metaworld_extrinsic_matrix,
        "intrinsic": Metaworld_intrinsic_matrix,
    },
    "rlbench": {
        "extrinsic": RLBench_extrinsic_matrix,
        "intrinsic": RLBench_intrinsic_matrix,
    },
    "maniskill": {
        "extrinsic": ManiSkill_extrinsic_matrix,
        "intrinsic": ManiSkill_intrinsic_matrix,
    }
}


def _resolve_benchmark_camera_key(benchmark: str) -> str:
    key = benchmark.lower()
    if key in CAMERA_CONFIGS:
        return key

    if "maniskill" in key:
        return "maniskill"
    if "metaworld" in key:
        return "metaworld"
    if "rlbench" in key:
        return "rlbench"

    raise ValueError(
        f"Unsupported benchmark '{benchmark}'. Available camera benchmarks: {list(CAMERA_CONFIGS.keys())}"
    )


def get_camera_params(benchmark: str, camera: str):
    """根据 benchmark 与 camera 名称获取内外参。"""
    key = _resolve_benchmark_camera_key(benchmark)
    cfg = CAMERA_CONFIGS[key]
    try:
        return cfg["intrinsic"][camera], cfg["extrinsic"][camera]
    except KeyError:
        raise ValueError(f"Camera {camera} not found in benchmark {benchmark}")

def rgbd_to_world_pointcloud(
    rgb: np.ndarray, 
    depth: np.ndarray, 
    mask: np.ndarray, 
    intrinsic: np.ndarray, 
    extrinsic: np.ndarray
):
    """
    将深度图转换为世界坐标系下的点云
    :param rgb: (H, W, 3) np.ndarray
    :param depth: (H, W) np.ndarray, 单位通常为米
    :param mask: (H, W) np.ndarray, bool 类型掩码
    :param intrinsic: (3, 3) 相机内参矩阵
    :param extrinsic: (4, 4) 相机到世界坐标系的变换矩阵
    :return: (N, 3) 世界坐标系下的点云
    """
    H, W = depth.shape
    
    # 1. 生成像素网格
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    
    # 2. 过滤掉非掩码区域的像素
    valid_mask = mask.astype(bool)
    z = depth[valid_mask]
    u = i[valid_mask]
    v = j[valid_mask]
    
    # 3. 反投影到相机坐标系 (x_cam = (u - cx) * z / fx)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 相机坐标系下的点 (N, 3)
    points_cam = np.stack([x, y, z], axis=1)
    
    # 4. 齐次坐标变换到世界坐标系
    ones = np.ones((points_cam.shape[0], 1))
    points_cam_homo = np.concatenate([points_cam, ones], axis=1)
    
    # points_world = extrinsic @ points_cam_homo^T
    points_world = (extrinsic @ points_cam_homo.T).T[:, :3]
    
    return points_world

def get_pointcloud_from_input(rgb, depth, mask, benchmark, camera):
    """
    集成函数：自动根据 benchmark 和 camera 名称获取内外参并转换
    """
    intrinsic, extrinsic = get_camera_params(benchmark, camera)
    return rgbd_to_world_pointcloud(rgb, depth, mask, intrinsic, extrinsic)