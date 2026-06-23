# 📰 Fake News Detection (AI/ML Task)

## 🎯 Project Goal  
This project builds a Fake News Detection model using Natural Language Processing (NLP) techniques to classify news articles as **real** or **fake** based on their text content.

---

## 🔬 Project Overview  

The project is structured both as a monolithic notebook and as a modular step-by-step pipeline.

- **Full End-to-End Notebook**: If you want the full whole notebook, refer to [Fake_News_Model.ipynb] which runs the entire workflow from start to finish.
- **Step-by-Step Pipeline**: If you want to build and run the pipeline step-by-step, run [notebooks/pipeline.ipynb]. Each cell executes a modular pipeline script under `src/`.

---

## 📁 Project Structure  

```
Fake-News-Model/
├── data/
│   ├── train.csv            # Training dataset CSV
│   └── test.csv             # Testing dataset CSV
├── notebooks/
│   └── pipeline.ipynb       # Runs each script cell-by-cell via %run
├── src/
│   ├── data_cleaning.py      # Stage 1: Text cleaning and preprocessing
│   ├── feature_extraction.py # Stage 2: TF-IDF feature extraction
│   ├── model_training.py     # Stage 3: Model training (Logistic Regression)
│   └── model_evaluation.py   # Stage 4: Evaluation metrics and plots
├── Fake_News_Model.ipynb    # Monolithic end-to-end notebook
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation (this file)
```

---

## 🚀 How to Run

### Setup Dependencies
Before running, install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Running the step-by-step Pipeline
1. Place the datasets inside the `data/` folder:
   - Save the training dataset as `data/train.csv`
   - Save the testing dataset as `data/test.csv`
2. Open and run the cells in [notebooks/pipeline.ipynb](file:///c:/Users/rohet/OneDrive/Documents/CS_WORK/Machine_Learning/Fake-News-Model/notebooks/pipeline.ipynb) sequentially.
   - Running Cell 1 imports `pandas` and loads the datasets automatically.
   - Subsequent cells execute the corresponding Python scripts under `src/` using Jupyter's `%run` magic command, maintaining variables in the notebook memory.

---

## 📊 Datasets  
The training and testing datasets can be downloaded from:
- **Training dataset**: [Google Drive Link](https://drive.google.com/file/d/1YpjG-avc4DA8TqSe5wfq8b76tXWaNzG1/view?usp=drive_link)
- **Testing dataset**: [Google Drive Link](https://drive.google.com/file/d/19LO-EbQdlZ7Uw6G8BdkPKE2ej4UsVK7V/view?usp=drive_link)
