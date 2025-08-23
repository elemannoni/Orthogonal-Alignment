import torch
import math
import copy

def lerp_models(model_a, model_b, t):
    """
    Interpolazione lineare tra model_a e model_b
    """
    new_model = copy.deepcopy(model_a) #per non modificare l'originale

    #interpolazione lineare
    for key in new_model.state_dict().keys():
        new_model.state_dict()[key].copy_((1 - t) * model_a.state_dict()[key] + t * model_b.state_dict()[key])

    return new_model

def slerp_models(model_a, model_b, t, eps=1e-7):
    """
    Interpolazione sferica tra model_a e model_b
    """
    new_model = copy.deepcopy(model_a)

    for key in new_model.state_dict().keys():
        a = model_a.state_dict()[key]
        b = model_b.state_dict()[key]

        #flatten dei tensori
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)

        #normalizzazione
        a_norm = a_flat / (a_flat.norm() + eps)
        b_norm = b_flat / (b_flat.norm() + eps)

        #angolo tra i due vettori
        dot = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
        theta = torch.acos(dot)

        if theta < eps:
            interpolated = (1 - t) * a + t * b
        else:
            sin_theta = torch.sin(theta)
            part_a = torch.sin((1 - t) * theta) / sin_theta
            part_b = torch.sin(t * theta) / sin_theta
            interpolated = part_a * a + part_b * b

        new_model.state_dict()[key].copy_(interpolated)

    return new_model
