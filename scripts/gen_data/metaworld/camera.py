import numpy as np
import metaworld
from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

def get_camera_intrinsic(model, camera_name, width, height):
    """
    计算相机的内参矩阵 K
    """
    # 获取相机 ID
    cam_id = model.camera(camera_name).id
    # 获取垂直视野 (fovy)，MuJoCo 存储的是角度
    fovy = model.cam_fovy[cam_id]
    # 计算焦距 f
    f = (height / 2) / np.tan(np.deg2rad(fovy) / 2)
    
    # 构建内参矩阵 K (假设主点在图像中心)
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    return K

def get_camera_extrinsic(model, data, camera_name):
    """
    计算相机的外参矩阵 [R | t] (World to Camera)
    """
    cam_id = model.camera(camera_name).id
    
    # 1. 相机在世界坐标系的位置
    pos = data.cam_xpos[cam_id]
    
    # 2. 相机在世界坐标系的旋转矩阵 (3x3)
    rot_matrix = data.cam_xmat[cam_id].reshape(3, 3)
    
    # 3. 坐标系转换：MuJoCo (-Z forward) -> OpenCV (+Z forward)
    # 绕 X 轴旋转 180 度
    flip_rot = R.from_euler('x', 180, degrees=True).as_matrix()
    rot_matrix_cv = rot_matrix @ flip_rot
    
    # 4. 构建外参 (World to Camera)
    # T_w2c = [R.T | -R.T @ pos]
    R_inv = rot_matrix_cv.T
    t_inv = -R_inv @ pos
    
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_inv
    extrinsic[:3, 3] = t_inv
    
    return extrinsic

# --- 主程序 ---

# 1. 选定一个 V3 环境名称
# 常见的如: 'pick-place-v3-goal-observable', 'reach-v3-goal-observable' 等
env_name = 'pick-place-v3-goal-observable'

if env_name not in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
    print(f"错误: {env_name} 不在 V3 环境列表中。")
    print("可用环境样例:", list(ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE.keys())[:5])
    exit()

print(f"正在初始化 V3 环境: {env_name}...")

# 2. 实例化环境
env_cls = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
env = env_cls(render_mode="rgb_array")

# V3 同样需要通过 ML1 或手动设置 task，否则 reset 可能会报错
# 这里使用简单的 reset 逻辑，MetaWorld 会随机初始化一个 task
obs, info = env.reset(seed=42)

# 3. 获取底层 MuJoCo 对象
# 在 Gymnasium 版本的 MetaWorld 中，必须使用 .unwrapped
mj_model = env.unwrapped.model
mj_data = env.unwrapped.data

# 4. 设定渲染分辨率 (用于计算内参)
width = 224
height = 224

# 5. 获取 corner 和 corner2 的参数
cameras = ["corner", "corner2", "topview"]

for cam_name in cameras:
    print(f"\n{'='*10} 相机: {cam_name} {'='*10}")
    
    try:
        # 获取内参
        K = get_camera_intrinsic(mj_model, cam_name, width, height)
        print("内参矩阵 K (Intrinsic):")
        print(np.array2string(K, precision=2, suppress_small=True))
        
        # 获取外参
        RT = get_camera_extrinsic(mj_model, mj_data, cam_name)
        print("\n外参矩阵 RT (Extrinsic - World to Camera):")
        print(np.array2string(RT, precision=4, suppress_small=True))
        
        # 物理位置参考
        cam_id = mj_model.camera(cam_name).id
        print(f"\n相机在世界坐标系的位置: {mj_data.cam_xpos[cam_id]}")
        
    except Exception as e:
        print(f"无法获取相机 '{cam_name}' 的参数: {e}")

# 6. 关闭环境
env.close()