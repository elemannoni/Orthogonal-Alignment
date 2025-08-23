import torch
from tqdm import tqdm

def train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler, epochs, tr_losses, val_losses):
  for epoch in range(epochs):
    model.eval()
    val_loss = 0
    with torch.no_grad():
      for batch in val_dataloader:
        x = batch["x"]
        y = batch["y"]
        out1 = model(x)

        loss1 = loss_fn(out1, y)

        loss = loss1
        val_loss += loss.item()

      val_losses.append(val_loss/len(val_dataloader))

    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader):
      x = batch["x"]
      y = batch["y"]
      out1= model(x)
      loss = loss_fn(out1, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
    tr_losses.append(train_loss/len(train_dataloader))

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {tr_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}")
