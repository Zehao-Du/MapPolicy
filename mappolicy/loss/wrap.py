import torch
from mappolicy.loss.eepose import compute_loss_cosine_similarity_quaternion
    
def smart_loss_func(preds, actions, lambda_map=1.0, lambda_physical=1.0):
    """
    preds: 可能是 Tensor (action), 
           也可能是 Tuple (action, aux_loss_map, aux_loss_2)
           也可能是 Dict {"action": ..., "map_loss": ..., "aux_loss": ...}
    lambda_map: 地图重构损失的权重 (对应你脚本中的 lambda_map)
    lambda_aux2: 第二个辅助损失的权重
    """
    
    # 1. 统一提取：将不同格式的 preds 转换为标准的变量
    if isinstance(preds, dict):
        pred_action = preds["action"]
        # 使用 .get() 防止键不存在时报错
        aux_loss_1 = preds.get("map_loss", torch.tensor(0.0, device=pred_action.device))
        aux_loss_2 = preds.get("aux_loss", torch.tensor(0.0, device=pred_action.device))
    
    elif isinstance(preds, (tuple, list)):
        pred_action = preds[0]
        aux_loss_1 = preds[1] if len(preds) > 1 else torch.tensor(0.0, device=pred_action.device)
        aux_loss_2 = preds[2] if len(preds) > 2 else torch.tensor(0.0, device=pred_action.device)
    
    else:  # 情况 B: 只有单一 Tensor
        pred_action = preds
        aux_loss_1 = torch.tensor(0.0, device=pred_action.device)
        aux_loss_2 = torch.tensor(0.0, device=pred_action.device)

    # 2. 计算各部分损失
    # 计算主任务 Loss (Action MSE)
    main_loss = torch.nn.functional.mse_loss(pred_action, actions)
    
    # 3. 应用权重并加总
    # 总损失 = Action损失 + lambda_map * 地图损失 + lambda_aux2 * 其他辅助损失
    total_loss = main_loss + (lambda_map * aux_loss_1) + (lambda_physical * aux_loss_2)

    # 4. 返回总损失和用于 Wandb 记录的标量字典
    return total_loss, {
        "loss/total": total_loss.item(),
        "loss/main_action": main_loss.item(),
        "loss/aux_map_raw": aux_loss_1.item() if torch.is_tensor(aux_loss_1) else aux_loss_1,
        "loss/aux_physical_raw": aux_loss_2.item() if torch.is_tensor(aux_loss_2) else aux_loss_2,
        "lambda/map": lambda_map
    }
    
def rlbench_loss_func(preds, actions, lambda_map=1.0, lambda_physical=1.0):
    """
    preds: 可能是 Tensor (action), 
           也可能是 Tuple (action, aux_loss_map, aux_loss_2)
           也可能是 Dict {"action": ..., "map_loss": ..., "aux_loss": ...}
    lambda_map: 地图重构损失的权重 (对应你脚本中的 lambda_map)
    lambda_aux2: 第二个辅助损失的权重
    """
    
    # 1. 统一提取：将不同格式的 preds 转换为标准的变量
    if isinstance(preds, dict):
        pred_action = preds["action"]
        # 使用 .get() 防止键不存在时报错
        aux_loss_1 = preds.get("map_loss", torch.tensor(0.0, device=pred_action.device))
        aux_loss_2 = preds.get("aux_loss", torch.tensor(0.0, device=pred_action.device))
    
    elif isinstance(preds, (tuple, list)):
        pred_action = preds[0]
        aux_loss_1 = preds[1] if len(preds) > 1 else torch.tensor(0.0, device=pred_action.device)
        aux_loss_2 = preds[2] if len(preds) > 2 else torch.tensor(0.0, device=pred_action.device)
    
    else:  # 情况 B: 只有单一 Tensor
        pred_action = preds
        aux_loss_1 = torch.tensor(0.0, device=pred_action.device)
        aux_loss_2 = torch.tensor(0.0, device=pred_action.device)

    # 2. 计算各部分损失
    # 计算主任务 Loss (Action)
    main_loss, _ = compute_loss_cosine_similarity_quaternion(pred_action, actions)
    
    # 3. 应用权重并加总
    # 总损失 = Action损失 + lambda_map * 地图损失 + lambda_aux2 * 其他辅助损失
    total_loss = main_loss + (lambda_map * aux_loss_1) + (lambda_physical * aux_loss_2)

    # 4. 返回总损失和用于 Wandb 记录的标量字典
    return total_loss, {
        "loss/total": total_loss.item(),
        "loss/main_action": main_loss.item(),
        "loss/aux_map_raw": aux_loss_1.item() if torch.is_tensor(aux_loss_1) else aux_loss_1,
        "loss/aux_physical_raw": aux_loss_2.item() if torch.is_tensor(aux_loss_2) else aux_loss_2,
        "lambda/map": lambda_map
    }