# 🎬 Movie Review Sentiment Analysis

A deep learning project that uses Natural Language Processing (NLP) to analyze movie reviews and classify them as **positive** or **negative**.

The project includes a neural-network-based sentiment classifier and an interactive Streamlit application for testing movie reviews.

---

## 📌 Project Overview

This project explores how deep learning and word embeddings can be used to perform sentiment analysis on movie reviews.

The project started with a trainable Keras embedding model using a **64-dimensional embedding**. As part of improving the model, the embedding dimension was increased to **128 dimensions**.

The current completed experiment focuses on comparing these two embedding configurations.

Further experimentation with **Word2Vec embeddings** is planned as the next stage of model improvement.

---

## 🎯 Problem

Movie reviews contain information about how viewers feel about a movie. However, machine-learning models cannot directly process raw natural-language text.

The goal of this project is to build a model that can:

- Process natural-language movie reviews
- Learn numerical representations of words
- Classify reviews as positive or negative
- Produce a confidence score for predictions

---

## 📊 Dataset

The project uses the **IMDB Dataset of 50,000 movie reviews**.

Each review belongs to one of two sentiment classes:

- `positive`
- `negative`

The dataset was cleaned before training, including the removal of duplicate reviews.

After removing duplicates, the dataset contained:

**49,582 reviews**

---

## 🧠 Model Approach

### Text Processing

The movie reviews are converted into numerical sequences using a Keras tokenizer.

The vocabulary is limited to the **10,000 most frequent words**.

Reviews are padded or truncated to a maximum length of **250 tokens** so that they can be processed by the neural network.

```python
max_length = 250
```

The data is divided into training and testing sets using an 80/20 split with a fixed random state:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## 🔤 Word Embeddings

The model uses a **trainable Keras Embedding layer**.

The embedding layer converts each word ID into a numerical vector that is learned during model training.

### Original Baseline

The original model used a 64-dimensional embedding:

```python
Embedding(
    input_dim=10000,
    output_dim=64
)
```

### 128D Experiment

For the assignment experiment, the embedding dimension was increased from **64 to 128**:

```python
Embedding(
    input_dim=10000,
    output_dim=128
)
```

The purpose of this experiment was to investigate whether giving each word a larger representation space would affect sentiment classification performance.

The rest of the model architecture and training setup were kept largely unchanged to make the comparison meaningful.

---

## 🏗️ Model Architecture

The current 128D experiment follows this architecture:

```text
Input Review
     ↓
Tokenization
     ↓
Padding / Truncation
     ↓
128-dimensional Word Embedding
     ↓
Global Average Pooling
     ↓
Dense Layer (64 units)
     ↓
Dropout (0.5)
     ↓
Sigmoid Output
     ↓
Positive / Negative
```

The final sigmoid layer produces a probability that is used to determine the predicted sentiment.

---

## 📈 Results

The original 64D model is used as the baseline for comparison with the 128D experiment.

| Metric | Original 64D | 128D Experiment |
|---|---:|---:|
| Accuracy | 87.66% | 86.97% |
| Precision | 90.99% | 83.06% |
| Recall | 83.69% | 93.01% |
| F1 Score | 87.19% | 87.75% |
| ROC-AUC | 95.10% | 94.90% |

### Results Interpretation

The 128D experiment produced a different precision-recall tradeoff compared with the original 64D model.

The 128D experiment had:

- Higher recall
- Slightly higher F1 score
- Lower precision
- Slightly lower accuracy
- Very similar ROC-AUC

Therefore, increasing the embedding dimension from 64 to 128 did **not** improve every evaluation metric.

This experiment demonstrates that increasing the size of an embedding representation can change model behavior, but a larger embedding does not automatically guarantee better overall performance.

> **Note:** The 128D results shown above represent the completed experimental run used for this project stage. Neural-network training can produce different results between runs because of random initialization and other training randomness.

---

## 🔎 Model Limitation Discovered

During testing with the Streamlit application, the model appeared to be more reliable on longer reviews than on some very short inputs.

For example:

```text
"I love the movie."
```

could sometimes receive an incorrect sentiment prediction.

This suggests that the current model may have difficulty extracting enough sentiment information from very short reviews.

One possible limitation is the use of:

```python
GlobalAveragePooling1D()
```

This layer averages the word representations across the review. It provides a simple representation of the overall text, but it does not explicitly model word order or relationships between words.

This observation provides motivation for experimenting with other word-representation and model architectures.

---

## 🚀 Next Experiment: Word2Vec

The next stage of the project will investigate **Word2Vec embeddings**.

Unlike the current trainable Keras embedding, Word2Vec learns word representations from the relationships between words and their surrounding context.

The planned workflow is:

```text
Training Reviews
      ↓
Tokenized Words
      ↓
Word2Vec
      ↓
Learned Word Vectors
      ↓
Embedding Matrix
      ↓
Sentiment Classifier
      ↓
Evaluation
```

The Word2Vec model will be evaluated using the same metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The goal is to determine whether an alternative word-representation approach can improve the current model, particularly when dealing with short reviews.

**Word2Vec experimentation and further model fine-tuning will continue after the current presentation.**

---

## 🖥️ Running the Application

### 1. Clone the Repository

```bash
git clone <repository-url>
cd movie-sentiment
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

### 3. Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```powershell
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 📁 Project Structure

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

---

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Natural Language Processing (NLP)
- Word Embeddings
- Git & GitHub

---

## 🔮 Future Improvements

- Experiment with Word2Vec embeddings
- Compare Word2Vec with trainable Keras embeddings
- Improve classification of short reviews
- Investigate sequence-aware architectures
- Tune model hyperparameters
- Improve prediction reliability
- Evaluate additional text-representation approaches
- Deploy the improved model to the Streamlit application

---

## 👩 Author

**Yvette Kazeneza**

Full-Stack Developer | UI/UX Designer | AI/ML Learner