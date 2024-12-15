# From https://github.com/vikram2000b/bad-teaching-unlearning

import torch
from sklearn.metrics import f1_score

from torch import nn
from torch.nn import functional as F
from training_utils import *


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100


def training_step(model, batch, device):
    images,  clabels = batch
    # print('batch',batch)
    # images, labels, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    clabels = clabels.squeeze().long()  # 是path的

    loss = F.cross_entropy(out, clabels)  # Calculate loss
    return loss


def validation_step(model, batch, device):
    model.eval()
    images,  clabels = batch
    # images, labels, clabels = batch
    # images = images.unsqueeze(0)  # 增加一个批次维度  jinxing dange shujuji yiwang de shihou women tianjiade

    images, clabels = images.to(device), clabels.to(device)
    # print( clabels.shape)
    # print(clabels)

    out = model(images)  # Generate predictions
    tensor11 = torch.randn(1, 1)
    if clabels.size() == tensor11.size() :
        # 如果是标量，增加一个维度并转换为 long 类型
        clabels = clabels.squeeze(0).long()

        # print(clabels.dim())
    else:
        # 如果不是标量，仅转换类型为 long
        clabels = clabels.squeeze().long()  # 注意，这里的 squeeze 实际上不会改变形状
        # print(clabels.dim())


    # clabels = clabels.squeeze().long()  # 是path的
    out_np = out.cpu().detach().numpy()
    clabels_np = clabels.cpu().detach().numpy()
    preds = torch.argmax(out, dim=1).cpu().detach().numpy()
    # print(out)
    # print(clabels)
    # print(out.shape, clabels.shape)
    loss = F.cross_entropy(out, clabels)  # Calculate loss
    acc = accuracy(out, clabels)  # Calculate accuracy
    f1 = f1_score(clabels_np, preds, average='weighted')  # You can choose 'macro', 'micro', etc.

    return {"Loss": loss.detach(), "Acc": acc, "F1": f1}


def validation_epoch_end(model, outputs):
    batch_losses = [x["Loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x["Acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    # 转换 F1 分数为张量并确保是张量
    batch_f1s = [torch.tensor(x["F1"], dtype=torch.float32) if not isinstance(x["F1"], torch.Tensor) else x["F1"]
                 for x in outputs]
    epoch_f1 = torch.stack(batch_f1s).mean()  # 组合并计算均值
    # batch_f1s = [x["F1"] for x in outputs]
    # epoch_f1 = torch.stack(batch_f1s).mean()  # Combine accuracies
    return {"Loss": epoch_loss.item(), "Acc": epoch_acc.item(), "F1": epoch_f1.item()}


def epoch_end(model, epoch, result):
    print(
        "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch,
            result["lrs"][-1],
            result["train_loss"],
            result["Loss"],
            result["Acc"],
        )
    )


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]
    return validation_epoch_end(model, outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit_one_cycle(
    epochs, model, train_loader, val_loader, device, lr=0.01, milestones=None
):
    torch.cuda.empty_cache()
    history = []

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    if milestones:
        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.2
        )  # learning rate decay
        warmup_scheduler = WarmUpLR(optimizer, len(train_loader))

    for epoch in range(epochs):
        if epoch > 1 and milestones:
            train_scheduler.step(epoch)

        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

            if epoch <= 1 and milestones:
                warmup_scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
    return history
