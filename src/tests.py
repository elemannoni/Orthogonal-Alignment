import torch
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def evaluate_model(model, dataloader):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
      for batch in dataloader:
        x = batch["x"]
        y = batch["y"]
        y_pred = model(x).to("cpu")
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss
  
def confusion_matrix(model, dataloader, device):
    model.eval()
    ys = []; preds = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            ys.append(y.cpu().numpy())
            preds.append(model(x).argmax(dim=1).cpu().numpy())
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    print(confusion_matrix(ys, preds))

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

  def cycle_consistency(model, Q_dict):
  for i, layer in enumerate(model.fc):
    if isinstance(layer, torch.nn.Linear):
      differenza = (Q_dict[i].T @ (Q_dict[i] @ model.fc[i].weight.data)) - model.fc[i].weight.data
      print(torch.norm(differenza))

