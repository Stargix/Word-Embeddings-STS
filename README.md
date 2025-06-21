# Semantic Textual Similarity (STS) for Catalan

This project implements, experiments with, and evaluates different model architectures for the task of semantic similarity between Catalan sentence pairs, using the STS-ca dataset from the AINA project. The goal is to compare various embedding techniques and models to determine which approach best fits the Semantic Textual Similarity (STS) task for Catalan.

---

## **Objectives**

- Develop and test different architectures to compute semantic similarity between pairs of Catalan sentences.
- Compare results from models based on pretrained embeddings (Word2Vec/FastText) and attention-based architectures.
- Analyze the impact of embedding dimensionality.
- Provide a quantitative analysis of the results obtained with different methods.

---

## **Project Structure**

- **Embedding Preparation**: Load and truncate pretrained Word2Vec/FastText embeddings for Catalan (`cc.ca.300.vec`).
- **Baseline Models**: Simple cosine similarity between sentence embeddings.
- **Model 1: Aggregated Embeddings**: Concatenate sentence vectors.
- **Model 2: Embedding Sequence**: Sequential model with attention mechanism.
- **Advanced Experimentation**: Comparison with other embeddings and techniques.
- **Conclusions and Observations**: Global comparison and discussion.
- **Training with TECLA Data**: Additional classification model training.

---

## **Dataset**

- **Source:** [projecte-aina/sts-ca](https://huggingface.co/datasets/projecte-aina/sts-ca)
- **Description:** Pairs of Catalan sentences with a similarity score (0-5).
- **Splits:**
  - Train: 2,073 examples
  - Validation: 500 examples
  - Test: 500 examples

---

## **Requirements**

- Python 3.8+
- Libraries:
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `scipy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `gensim`
  - `datasets`
- **Pretrained model:** `cc.ca.300.vec` (FastText, Catalan) available at [fasttext.cc](https://fasttext.cc/docs/en/crawl-vectors.html).

Install dependencies:
