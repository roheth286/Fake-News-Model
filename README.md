# 📰 Fake News Detection (AI/ML Task)

## 🎯 Project Goal  
This project builds a Fake News Detection model using Natural Language Processing (NLP) techniques to classify news articles as **real** or **fake** based on their text content. The entire workflow is implemented in a single Jupyter notebook, from loading the training and test datasets to text preprocessing, feature extraction, model training, and evaluation.

---

## 🔬 Project Overview  

The notebook is organized into four main stages:

1. **Data Preparation**
   - Load the Fake News **training** and **testing** datasets.
   - Handle missing values and clean the text/data.
   - Encode any categorical features and scale numerical data if needed.
   - Align features and target labels for model training.

2. **Feature Extraction**
   - Convert raw text into numerical features using NLP methods such as:
     - **TF‑IDF**
     - **Count Vectorizer**
     - Or other embeddings (depending on the implementation in the notebook).

3. **Machine Learning Model Training**
   - Train one or more baseline classification models for fake news detection.
   - Apply basic feature selection and tuning to improve performance.

4. **Model Evaluation**
   - Evaluate models using:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1‑score**
   - Generate and inspect a **classification report** to analyze predictions and per‑class performance.

---

## 📊 Datasets  

- The notebook includes **two dataset links**:
  - One for the **training dataset**
  - One for the **testing dataset**
- These links are provided directly in the notebook.

> You must download both datasets **locally** (e.g., to your Downloads folder) using the links inside the notebook, then upload them when prompted.

---

## 📁 Project Structure  

- `README.md` → Project description and usage instructions (this file).  
- `fake_news_detection.ipynb` (or similar) → Main notebook containing:
  - Dataset download links  
  - Data preparation and feature extraction  
  - Model training and evaluation  

---

## 🚀 How to Run (Google Colab Recommended)  

1. **Open the notebook in Colab**
   - Upload the `.ipynb` file to Google Drive and open it with **Google Colaboratory**,  
   - Or open it from GitHub with an “Open in Colab” link.

2. **Download and import the datasets**
   - In the notebook, go to the section where the **training** and **testing** dataset links are provided.  
   - Click each link and download both CSV files **to your local machine**.  
   - Find the **“Import Dataset”** cell in the notebook and run it:
     - When prompted, upload the **training** file.
     - When prompted (if in a separate cell), upload the **testing** file.  
   - After uploading, the notebook will load these files into the Colab environment for processing.

3. **Run the notebook cells in order**
   - Run the **Data Preparation** section to clean and preprocess the data.  
   - Run the **Feature Extraction** section to transform text into numerical vectors (TF‑IDF / Count Vectorizer / embeddings).  
   - Run the **Model Training** section to fit the fake news classifiers.  
   - Run the **Model Evaluation** section to view metrics and the classification report.

---

## 📄 Notes  

- All core logic is contained inside the notebook; any extra Python packages (such as `pandas`, `scikit‑learn`, `numpy`, and text feature extraction libraries) are installed via `pip` commands in the notebook if needed.  
- Always ensure you upload the correct **train** and **test** CSV files when the **Import Dataset** cell prompts for file uploads.
