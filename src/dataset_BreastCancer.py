import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random
from sklearn.preprocessing import LabelEncoder

class SpecDataset(torch.utils.data.Dataset):
  def __init__(self, pairs):
    super().__init__()
    self.pairs = pairs

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    return {"x":self.pairs[idx][0], "y":self.pairs[idx][1]}


def create_dataloaders(seed=2019, batch_size=32):
  """
  Crea i dataloader dal datest BreastCancer per due modelli
  """
  dataset = load_dataset("scikit-learn/breast-cancer-wisconsin")
  data = dataset["train"]

  #Per codificare M e B in 0 e 1
  y = [x["diagnosis"] for x in data]
  le = LabelEncoder()
  y = le.fit_transform(y)  # M -> 1, B -> 0
  
  feature_cols = [f for f in data.column_names if f not in ["id", "diagnosis", "Unnamed: 32"]]
  X = [[x[f] for f in feature_cols] for x in data]
  
  x = torch.tensor(X, dtype=torch.float32)
  y = torch.tensor(y, dtype=torch.long)
  
  pairs = []
  for i in range(len(y)):
    pairs.append((x[i], y[i]))

  torch.manual_seed(seed)
  pairs  = random.shuffle(pairs)
  test = pairs[int(len(pairs)*0.9):]
  
  #test dataloader 
  test_dataset = SpecDataset(test)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
  
  pairs = pairs[:int(len(pairs)*0.9)]

  #val e train dataloader per il primo modello
  #torch.manual_seed(seed1)
  pairs1 = random.sample(pairs, k = len(pairs))
  train = pairs1[:int(len(pairs1)*0.8)]
  val = pairs1[int(len(pairs1)*0.8):]
  
  train_dataset1 = SpecDataset(train)
  val_dataset1 = SpecDataset(val)

  g1 = torch.Generator()
  g1.manual_seed(seed) #per fissare il seed anche nello shuffle del dataloader

  train_dataloader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=batch_size, shuffle=True, generator=g1, num_workers=2)
  val_dataloader1 = torch.utils.data.DataLoader(val_dataset1, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

  #val e train dataloader per il secondo modello
  #torch.manual_seed(seed2)
  pairs2 = random.sample(pairs, k = len(pairs))
  train = pairs2[:int(len(pairs2)*0.8)]
  val = pairs2[int(len(pairs2)*0.8):int(len(pairs2)*0.9)]
  
  train_dataset2 = SpecDataset(train)
  val_dataset2 = SpecDataset(val)

  g2 = torch.Generator()
  g2.manual_seed(seed)

  
  train_dataloader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=batch_size, shuffle=True, generator=g2, num_workers=2)
  val_dataloader2 = torch.utils.data.DataLoader(val_dataset2, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

  return (train_dataloader1, val_dataloader1), (train_dataloader2, val_dataloader2), test_dataloader
