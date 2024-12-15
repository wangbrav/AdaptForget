import torch
import torch.nn.functional as F

def calculate_kl_divergence(smodel,u, smodel_base2, data_loader, device):
    model_1 = smodel.to(device)

    model_2 =smodel_base2.to(device)
    model_2.load_state_dict(u)


    model_1.eval()
    model_2.eval()

    kl_divergences = []

    for images, _ in data_loader:  
        images = images.to(device)  

        # 进行预测
        output_1 = model_1(images)
        output_2 = model_2(images)


        kl_div = F.kl_div(output_1, torch.exp(output_2), reduction='batchmean')

        kl_divergences.append(kl_div.item())

    average_kl_div = sum(kl_divergences) / len(kl_divergences)

    return max(0.0, average_kl_div)
