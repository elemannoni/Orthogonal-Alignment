# Orthogonal-Alignment
In questo progetto si cerca di estendere l'allineamento dei pesi di due modelli con la stessa architettura, proposto nel paper Git: Re-basin tramite le matrici di permutazione, all'utilizzo di matrici ortogonali

# Struttura
│── src/ #file Python con funzioni e classi<br>
│ ├── datasets.py<br>
│ ├── models.py <br>
│ ├── train.py<br>
│ ├── tests.py<br>
│ ├── merge.py<br>
│ ├── analysis.py<br>
│── requirements.txt # librerie Python<br>
│── README.md # documentazione<br>

Link per aprire i notebook in Colab:<br>
Main notebook: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elemannoni/Orthogonal-Alignment/blob/main/Orthogonal_alignment_notebook.ipynb)

**src**<br>
  • dataset.py =  definzione della classe dei dataset, download dei dataset e creazione dei DataLoader<br>
  • models.py = definzione della classe dei modelli<br>
  • train.py = funzione per il loop di training<br>
  • tests.py = funzione per valutare la loss di test, funzioni per chek sulla correttezza del modello<br>
  • merge.py = funzioni per SLERP e LERP<br>
  • analysis.py = analisi non linearità, cycle consistency, geometria dello spazio<br>


