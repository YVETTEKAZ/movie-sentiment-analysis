#  Movie Review Sentiment Analysis

A deep learning project that analyzes movie reviews and predicts whether the review expresses a **positive** or **negative** sentiment.

##  Project Overview

This project uses Natural Language Processing (NLP) and a neural network to classify movie reviews based on their sentiment.

The project started with a trainable word-embedding model using a 64-dimensional embedding. As part of improving the project, the embedding representation was extended to 128 dimensions.

The project is also deployed as an interactive Streamlit application where users can enter a movie review and receive a sentiment prediction with a confidence score.

##  Problem

Movie reviews contain useful information about how viewers feel about a movie. However, computers cannot directly understand raw text.

The goal of this project is to build a model that can:

- Process natural-language movie reviews
- Learn useful representations of words
- Classify reviews as positive or negative
- Provide a confidence score for predictions

##  Dataset

The project uses the **IMDB Dataset of 50,000 movie reviews**.

Each review has one of two sentiment labels:

- `positive`
- `negative`

The dataset was cleaned before training, including removing duplicate reviews.

After duplicate removal, the dataset contained **49,582 reviews**.

##  Model Approach

### Text Processing

The reviews are converted into numerical representations using tokenization.

The vocabulary is limited to the most frequent 10,000 words:

```python
Embedding(
    input_dim=10000,
    output_dim=128
)
```

Reviews are then padded to the same sequence length so they can be processed by the neural network.

### Word Embeddings

The model uses a **trainable Keras Embedding layer**.

Each word is represented by a numerical vector that the model learns during training.

The current experiment uses:

```text
Vocabulary size: 10,000
Embedding dimension: 128
```

The embedding dimension was increased from **64 to 128** as an experiment to give the model a larger representation space for learning word patterns.

### Model Architecture

```text
Input review
     ↓
Tokenization
     ↓
Padding
     ↓
128-dimensional Word Embedding
     ↓
Global Average Pooling
     ↓
Dense layer (64 units)
     ↓
Dropout (0.5)
     ↓
Sigmoid output
     ↓
Positive / Negative
```

## Baseline Results

The original model used a 64-dimensional trainable embedding.

The improved experiment used a 128-dimensional trainable embedding.

| Metric | 64D Model | 128D Model |
|---|---:|---:|
| Accuracy | 87.66% | 86.97% |
| Precision | 90.99% | 83.06% |
| Recall | 83.69% | 93.01% |
| F1 Score | 87.19% | 87.75% |
| ROC-AUC | 95.10% | 94.90% |

> **Note:** The 128D values represent one experimental training run. Neural-network training can produce slightly different results between runs because of random initialization and training behavior. Future experiments will use controlled random seeds for more reproducible comparisons.

### Observations

Increasing the embedding dimension from 64 to 128 produced a different precision-recall tradeoff.

The 128D experiment:

- achieved higher recall;
- achieved a slightly higher F1 score;
- had lower precision;
- had slightly lower accuracy;
- had a very similar ROC-AUC.

Therefore, increasing the embedding dimension alone did not improve every evaluation metric.

##  Model Limitation Discovered

During testing with the Streamlit application, the model appeared to perform more reliably on longer reviews than on very short reviews.

For example, a longer review containing several sentiment-related words can provide the model with more information than a short input such as:

```text
"I love the movie."
```

The current architecture also uses `GlobalAveragePooling1D`, which averages the word representations across the review. This provides a simple representation of the review but does not explicitly model word order.

This limitation motivates the next experiment using **Word2Vec embeddings**.

##  Next Experiment: Word2Vec

The next stage of the project will investigate Word2Vec embeddings.

Instead of relying only on the embedding representation learned directly by the classifier, Word2Vec will be trained on the movie-review training data to learn word representations from word-context relationships.

The Word2Vec experiment will then be compared with the current 128D baseline using the same evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

##  Running the Application

### 1. Clone the repository

```bash
git clone <repository-url>
cd movie-sentiment
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```powershell
python -m streamlit run app.py
```

The application should open in your browser at:

```text
http://localhost:8501
```

##  Project Structure

```text
movie-sentiment/
│
├── app.py
├── movie-review-sentiment-analysis.ipynb
├── sentiment_model.keras
├── tokenizer.json
├── requirements.txt
├── .gitignore
└── README.md
```

##  Technologies

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- NLP
- Word Embeddings
- Git & GitHub

##  Future Improvements

- Experiment with Word2Vec embeddings
- Compare Word2Vec with the current trainable embeddings
- Improve performance on short reviews
- Investigate sequence-aware architectures
- Tune model hyperparameters
- Improve prediction confidence
- Deploy the improved model to the Streamlit application

## 👩 Author

**Yvette Kazeneza**

Full-Stack Developer | UI/UX Designer | AI/ML Learner