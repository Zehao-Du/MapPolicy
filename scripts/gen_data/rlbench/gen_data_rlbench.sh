#!/bin/bash

export MUJOCO_GL=egl

# 2. 确保 Python 能找到你的项目包 (防止 ModuleNotFoundError)
# $(pwd) 代表当前目录，确保你在项目根目录运行此脚本
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 设置变量
RLBENCH_DATA_ROOT="/data2/zehao/StructureMap3D/Data/RLBench_224"
DATASET_ROOT="/data2/zehao/StructureMap3D/data/rlbench"

# 获取代码根目录 (等同于 Python 中的 pathlib.Path(__file__).resolve().parent.parent)
# 假设该脚本放在 Python 脚本原来的位置
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CODE_ROOT="$(dirname "$SCRIPT_DIR")"
TOOL_PATH="$CODE_ROOT/scripts/gen_data_rlbench.py"

# 定义任务列表
TASKS=(
    # "close_box"
    # "put_rubbish_in_bin"
    # "close_laptop_lid"
    "water_plants"
    # "unplug_charger"
    # "toilet_seat_down"
)

# 循环执行任务
for TASK in "${TASKS[@]}"; do
    echo "Running task: $TASK"

    xvfb-run -a python "$TOOL_PATH" \
        --rlbench-data-root "$RLBENCH_DATA_ROOT" \
        --task-name "$TASK" \
        --camera-name "front" \
        --point-cloud-camera-names "front" \
        --num-points "1024" \
        --rotation-representation "quaternion" \
        --image-size "224" \
        --num-episodes "120" \
        --only-keypoints \
        --save-dir "$DATASET_ROOT" \
        --quiet

    # 检查上一个命令是否成功
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing task: $TASK"
        exit 1
    fi
done

echo "All tasks completed."