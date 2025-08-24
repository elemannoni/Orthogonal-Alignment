import torch
import copy

def Q_min_quadrati(A, B, alpha=False):
    """
    trova Q che minimizza ||A - QB||_F
    """
  
    M = B.T @ A          # dimensione: (d, d)
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    Q = U @ Vt               # dimensione (d, d)
    return Q

def activation_matrix(model, dataloader, layer_num, device="cpu"):
    """
    Calcola la matrici di attivazione per un determinato layer
    """
    model.eval()
    activations_list = []
    with torch.no_grad():
      for batch in dataloader:
        x = batch["x"].to(device)
        x = x.to(device)
        for i, layer in enumerate(model.fc): #devo far passare l'input per tutti i layer fino a quello richiesto
          x = x.view(x.size(0), -1)
          x = layer(x)
          if i == layer_num:  #si ferma al layer richiesto
            break
        activations_list.append(x)
      A = torch.cat(activations_list, dim=0) #dimensioni: (N,d)
    return A

def apply_Q(model, Q_dict):
    """
    applica Q per ogni layer senza considerare i precedenti
    """
  
    al_model = copy.deepcopy(model)
    prev_Q = None

    for i, layer in enumerate(al_model.fc):
      if isinstance(layer, torch.nn.Linear): #applico ai layer linear
        Q_curr = Q_dict[i].to(layer.weight.device)
        W = layer.weight.data
        b = layer.bias.data if layer.bias is not None else None

        if prev_Q is None:
          W_al= Q_curr @ W
        else:
          W_al = Q_curr @ W @ prev_Q.T
        b_al =  Q_curr @ b if b is not None else None

        #Aggiornamento dei layer
        layer.weight.data = W_al
        layer.bias.data = b_al

        prev_Q = Q_curr

    return al_model

def apply_Q_layerwise(model1, model2, train_dataloader1, train_dataloader2, alpha=0.7, m=True):
    """
    Applica  Q considerando anche come sono stati modificati precedentemente gli altri layer
    """
    if m:
      m1 = model1
      al_model = copy.deepcopy(model2)
    else:
      m1 = model2
      al_model = copy.deepcopy(model1)
        
    x = next(iter(train_dataloader2))["x"].float()
    prev_Q = None

    for i, layer in enumerate(al_model.fc):
      if isinstance(layer, torch.nn.Linear): #applico ai layer linear
        A, B = activation_matrix(m1, train_dataloader1, i), activation_matrix(al_model, train_dataloader2, i)
        Q = Q_min_quadrati(A, B).to(layer.weight.device)
        if prev_Q is None:
          layer.weight.data = alpha * (Q @ layer.weight.data) + (1 - alpha) * layer.weight.data
        else:
          layer.weight.data = alpha * (Q @ layer.weight.data  @ prev_Q.T) + (1 - alpha) * layer.weight.data
        if layer.bias.data is not None:
          layer.bias.data = alpha * (Q @ layer.bias.data) + (1 - alpha) * layer.bias.data
        prev_Q = Q

        diff = torch.norm(A - B @ Q, 'fro')
        ortho_err = torch.norm(Q.T @ Q - torch.eye(Q.shape[0], device=Q.device)).item()
        det = torch.det(Q).item()
        print(f'Layer {i} matching error: {diff.item()}')
        print(f"Layer {i} ortho_err: {ortho_err:.2e} -> Q is orthogonal: {torch.allclose(Q.T @ Q, torch.eye(Q.shape[0]), atol=1e-5)}")
      x = layer(x)

    return al_model
