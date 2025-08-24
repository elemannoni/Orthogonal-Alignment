import torch
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from .orthogonal_alignment import activation_matrix, apply_Q_layerwise


def evaluate_model(model, dataloader):
    """
    valuta il modello sul dataloader con la Cross Entropy
    """
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
      for batch in dataloader:
        x = batch["x"]
        y = batch["y"]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss

def misalignment_ReLU(model1, al_model, dataloader, Q_list, linear_layers):
    model1.eval()
    al_model.eval()
    misalignments = []
    before_ReLU = []

    for idx, layer_idx in enumerate(linear_layers):
        # Matrici di attivazione
        A = activation_matrix(model1, dataloader, layer_idx)  # [N, d]
        B = activation_matrix(al_model, dataloader, layer_idx)  # [N, d]

        Q = Q_list[layer_idx]  # [d, d]
        AQ = A @ Q  # Applichiamo Q alla matrice di attivazione del prima modello prima della ReLU

        before_reLU = (torch.norm(AQ-B, p = "fro")/A.shape[0])

        # Applichiamo le ReLU
        AQ_relu = F.relu(AQ)
        B_relu = F.relu(B)

        # Misalignment: differenza tra i due layer una volta applicata la ReLU
        misalignment = torch.norm(AQ_relu - B_relu, p="fro") / A.shape[0]
        misalignments.append(misalignment.item())
        print(f"Layer {layer_idx} misalignment before ReLU: {before_reLU:.4f}")
        print(f"Layer {layer_idx} misalignment after ReLU: {misalignment.item():.4f}")

def cycle_consistency_ABA(model_A, model_B, train_dataloaderA, train_dataloaderB):
    model_AB, Q_dict1 = apply_Q_layerwise(model_B, model_A, train_dataloaderB, train_dataloaderA, 0.8) #allineo i pesi di A a B
    model_ABA, Q_dictBA = apply_Q_layerwise(model_A, model_AB, train_dataloaderA, train_dataloaderB, 0.8) #allineo i pesi del modello ottenuto ad A
  
    for i, layer in enumerate(model_A.fc):
        if isinstance(layer, nn.Linear):
          W_A = layer.weight.data
          W_ABA = model_ABA.fc[i].weight.data
          diff = torch.norm(W_A - W_ABA)
          print(f"Layer {i} ABA error: {diff.item()}")

def cycle_consistency_ABC(model_A, model_B, model_C, train_dataloaderA, train_dataloaderB, train_dataloaderC):
    model_AB, Q_dict_AB = apply_Q_layerwise(model_B, model_A, train_dataloaderB, train_dataloaderA, 0.8) #allineo i pesi di A a B
    model_ABC, Q_dict_BC = apply_Q_layerwise(model_C, model_AB, train_dataloaderC, train_dataloaderB, 0.8) #allineo il modello ottenuto a C
    model_AC, Q_dict_AC = apply_Q_layerwise(model_C, model_A, train_dataloaderC, train_dataloaderA, 0.8) #allineo i pesi di A a C
    
    for i, layer in enumerate(model_A.fc):
      if isinstance(layer, nn.Linear):
        W_ABC = model_ABC.fc[i].weight.data
        W_AC = model_AC.fc[i].weight.data
        diff = torch.norm(W_ABC - W_AC)
        print(f"Layer {i} ABC vs AC error: {diff.item()}")

def misalignment_ReLU(model1, al_model, dataloader, Q_list):
    model1.eval()
    al_model.eval()
    misalignments = []
    before_ReLU = []

    linear_layers = [i for i, layer in enumerate(model1.fc) if isinstance(layer, torch.nn.Linear)]

    for idx, layer_idx in enumerate(linear_layers):
        layer_idx += 1
        A = activation_matrix(model1, dataloader, layer_idx) 
        B = activation_matrix(al_model, dataloader, layer_idx) 

        Q = Q_list[f"fc{layer_idx}"]
        AQ = A @ Q  #applico Q alla matrice di attivazione prima della ReLU

        before_reLU = (torch.norm(AQ-B, p = "fro")/A.shape[0])

        #applico le ReLU
        AQ_relu = F.relu(AQ)
        B_relu = F.relu(B)

        # Misalignment: differenza tra i due layer una volta applicata la ReLU
        misalignment = torch.norm(AQ_relu - B_relu, p="fro") / A.shape[0]
        misalignments.append(misalignment.item())
        print(f"Layer {layer_idx} misalignment before ReLU: {before_reLU:.4f}")
        print(f"Layer {layer_idx} misalignment after ReLU: {misalignment.item():.4f}")

    return before_ReLU, misalignments
