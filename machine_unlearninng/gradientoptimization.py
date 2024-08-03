import torch.optim as optim


def compute_and_update_weights(loss,smodel,weights):
    """
    计算损失的梯度并更新模型的权重。

    :param loss: 计算出的损失值
    :param optimizer: torch.optim优化器实例
    """
    # 清除旧的梯度
    print("compute and update weights")
    smodel.load_state_dict(weights)
    optimizer = optim.Adam(smodel.parameters(), lr=0.001)

    optimizer.zero_grad()

    # 计算新的梯度
    loss.backward()

    # 根据梯度更新模型的权重
    optimizer.step()