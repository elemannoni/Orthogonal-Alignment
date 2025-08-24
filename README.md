# Orthogonal-Alignment
In questo progetto si cerca di estendere l'allineamento dei pesi di due modelli con la stessa architettura, proposto nel paper *Git: Re-basin* tramite le matrici di permutazione, all'utilizzo di matrici ortogonali

# Struttura
│── src/ <br>
│ ├── dataset_BreastCancer.py<br>
│ ├── models.py <br>
│ ├── train.py<br>
│ ├── test.py<br>
│ ├── merge.py<br>
│ ├── test_merge.py<br>
│ ├── orthogonal_alignment.py<br>
│── Orthogonal_Alignment_notebook.ipynb<br>
│── README.md<br>

Link per aprire i notebook in Colab:<br>
Main notebook: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elemannoni/Orthogonal-Alignment/blob/main/Orthogonal_Alignment_notebook.ipynb)

**src**<br>
  • dataset_BreastCancer.py =  definzione della classe dei dataset, download dei dataset e creazione dei DataLoader<br>
  • models.py = definzione della classe dei modelli<br>
  • train.py = funzione per il loop di training, analisi non linearità e cycle consistency<br>
  • test.py = funzione per valutare la loss di test, funzioni per chek sulla correttezza del modello<br>
  • merge.py = funzioni per SLERP e LERP<br>
  • test_merge.py = plot confronto interpolazione<br>
  • orthogonal_alignment.py = funzioni che implementano l'allineamento ortogonale<br>
