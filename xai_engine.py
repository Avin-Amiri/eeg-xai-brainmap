import torch
import numpy as np
from captum.attr import IntegratedGradients

def get_channel_attributions(model, x_tensor, target_class, device='cpu'):
    model.eval()
    ig = IntegratedGradients(model)
    x_tensor = x_tensor.to(device)
    baseline = torch.zeros_like(x_tensor).to(device)
    
    attributions, _ = ig.attribute(
        x_tensor,
        baselines=baseline,
        target=target_class,
        return_convergence_delta=True
    )

    importance = attributions.squeeze().abs().mean(dim=-1).detach().cpu().numpy()
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    return importance
