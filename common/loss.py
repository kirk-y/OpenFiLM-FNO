import torch
from torch import nn
from torch.nn import functional as F
from neuralop.losses import H1Loss, LpLoss

class LossMse(nn.Module):
    def __init__(self):
        super(LossMse, self).__init__()
        self.mse_loss = nn.MSELoss()
        

    def forward(self, predictions: torch.Tensor, batch) -> torch.Tensor:
        targets = batch['w_new']
        assert torch.isinf(predictions).sum() == 0, f"predictions contains inf: {predictions}"
        assert torch.isinf(targets).sum() == 0, f"targets contains inf: {targets}"
        assert torch.isnan(predictions).sum() == 0, f"predictions contains nan: {predictions}"
        assert torch.isnan(targets).sum() == 0, f"targets contains nan: {targets}"
        losses = self.mse_loss(predictions, targets)
        assert torch.isnan(losses).sum() == 0, f"Loss contains nan: {losses}"
        return losses

class FSCLoss(nn.Module):
    """
    流固耦合问题专用损失函数 (Fluid-Solid Coupling Loss)
    
    该损失函数结合了多种物理约束，包括:
    1. 基础MSE损失: 预测裂缝宽度与真实裂缝宽度之间的差异
    2. 增量损失: 裂缝宽度变化量的损失
    3. 能量损失: 基于流固耦合能量守恒的损失
    4. 物理一致性损失: 确保预测满足流固耦合的基本物理规律
    """
    def __init__(self, weights=[1.0, 0.1, 0.1, 0.01]):
        super(FSCLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lp_loss = LpLoss()
        self.weights = weights  # 各项损失的权重

    def forward(self, w_pred: torch.Tensor, batch) -> torch.Tensor:
        n = batch['n'][0]
        w_new = batch['w_new'][:, :n]
        w_old = batch['w_old'][:, :n]
        A = batch['A'][:, :n, :n]
        b = batch['b'][:, :n]
        
        # 1. 基础MSE损失
        mse_loss = self.mse_loss(w_pred, w_new)
        
        # 2. 增量损失 (裂缝宽度变化量)
        increment_pred = w_pred - w_old
        increment_true = w_new - w_old
        increment_loss = self.lp_loss(increment_pred, increment_true).mean()
        
        # 3. 能量损失 (基于流固耦合的能量守恒)
        energy_pred = 0.5 * torch.einsum('bi,bij,bj->b', w_pred, A, w_pred) - torch.einsum('bi,bi->b', w_pred, b)
        energy_true = 0.5 * torch.einsum('bi,bij,bj->b', w_new, A, w_new) - torch.einsum('bi,bi->b', w_new, b)
        energy_loss = self.mse_loss(energy_pred, energy_true)
        
        # 4. 物理一致性损失 (裂缝宽度应为非负值)
        physics_loss = torch.relu(-w_pred).mean()
        
        # 组合损失
        total_loss = (self.weights[0] * mse_loss + 
                     self.weights[1] * increment_loss + 
                     self.weights[2] * energy_loss + 
                     self.weights[3] * physics_loss)
        
        return total_loss

class CompositeLoss(nn.Module):
    def __init__(self):
        super(CompositeLoss, self).__init__()
        self.fno_loss = LpLoss()
        # self.fno_loss = H1Loss()
    
    def forward(self, w_pred: torch.Tensor, batch, weight:list=[1, 1, 1]) -> torch.Tensor:
        if 'pred' in batch:
            # New mode: w_pred is normalized increment, target is normalized increment
            batch_size = batch['pred'].shape[0]
            losses = []
            for i in range(batch_size):
                n_i = int(batch['n'][i].item()) if isinstance(batch['n'], torch.Tensor) else int(batch['n'][i])
                pred_i = w_pred[i, :n_i]
                target_i = batch['pred'][i, :n_i]
                losses.append(self.fno_loss(pred_i, target_i))
            return torch.stack(losses).mean()
		
        batch_size = batch['w_new'].shape[0]
        w_new = batch['w_new']
        w_old = batch['w_old']
        losses = []
        for i in range(batch_size):
            n_i = int(batch['n'][i].item()) if isinstance(batch['n'], torch.Tensor) else int(batch['n'][i])
            pred_i = w_pred[i, :n_i]
            w_new_i = w_new[i, :n_i]
            w_old_i = w_old[i, :n_i]
            increment = self.increment_loss(pred_i, w_new_i, w_old_i)
            losses.append(increment)
        # return mse * weight[0]#  + hsd * weight[1] + increment * weight[2]
        return torch.stack(losses).mean()
        # return mse
        # return mse + increment
    
    def mse_loss(self, w_pred, w_true):
        return F.mse_loss(w_pred, w_true)
    
    def HausdorffDistance_loss(self, w_pred, w_true, n, threshold=1e-5):
        mask = torch.zeros_like(w_pred)
        for i in range(n.shape[0]):  # 有效单元设为 1
            mask[i, :n[i]] = 1
        # 提取非尖端区域的活跃裂缝点
        pred_points = torch.where((w_pred > threshold) & (mask == 1), w_pred, torch.tensor(0.))
        true_points = torch.where((w_true > threshold) & (mask == 1), w_true, torch.tensor(0.))
        # 计算双向最大距离
        if pred_points.size(0) == 0 or true_points.size(0) == 0:
            return 0.0
        dist1 = torch.cdist(pred_points, true_points).min(dim=1)[0].max()
        dist2 = torch.cdist(true_points, pred_points).min(dim=1)[0].max()
        return torch.max(dist1, dist2)
    
    # 增量损失
    def increment_loss(self, w_pred, w_new, w_old):
        increment1 = (w_pred).squeeze()
        increment2 = (w_new - w_old).squeeze()
        increment = self.fno_loss(increment1, increment2)
        # increment = self.fno_loss(w_pred, w_true)
        return increment.mean()

    def energy_loss(self, w_pred, A, w_true):
        energy_pred = 0.5 * torch.einsum('bi,bij,bj->b', w_pred, A, w_pred)
        energy_old = 0.5 * torch.einsum('bi,bij,bj->b', w_true, A, w_true)
        return torch.abs(energy_pred - energy_old).mean()


def loss_fn(loss_type: str='mse') -> nn.Module:
    # return LossMse()
    if loss_type == 'fsc':
        return FSCLoss()
    return CompositeLoss()