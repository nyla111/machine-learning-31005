# Study of Logistic Regression and Model Complexity

## Overview
This repository contains a Jupyter Notebook developed for the course **Advanced Data Analytics Algorithms and Machine Learning** at the University of Technology Sydney. It documents a **progressive study of logistic regression**, implemented from scratch and extended across multiple weeks. The project starts with binary logistic regression, expands to multi-class classification using softmax, and concludes with a **controlled comparison against a simple neural network** to examine when additional model complexity helps - and when it does not.

The work emphasizes **theoretical grounding, clean implementation, and task-appropriate evaluation**, rather than treating models as black boxes.

---

## Project Objectives
- Understand logistic regression as a **model family and hypothesis space**
- Implement binary and multi-class logistic regression **from scratch**
- Design a clear training and deployment interface (inputs, outputs, objectives)
- Evaluate models using metrics aligned with **real-world decision costs**
- Compare logistic regression with a small neural network to study the **bias-variance trade-off**

---

## Dataset
- **Breast Cancer Wisconsin (Diagnostic) dataset** (scikit-learn)
- 569 samples, 30 numeric features
- Binary labels: malignant (0) vs benign (1)
- Chosen for interpretability, moderate size, and real-world relevance

Features are standardized using z-score normalization, and data is split using an 80/20 stratified train–test split.

---

## Repository Structure & Deliverables
- **`ML_Week1.ipynb` - `ML_Week8.ipynb`**

These notebooks represent cumulative progress:
- Early weeks: binary logistic regression, loss functions, gradient descent
- Middle weeks: numerical stability, vectorization, evaluation metrics
- Later weeks: multi-class softmax regression and threshold tuning
- Final week: implementation and comparison with a simple neural network

Together, they form the complete deliverable and learning trajectory.

---

## Models Implemented
### Logistic Regression
- Binary classification with sigmoid activation
- Trained using cross-entropy loss and gradient descent
- Implemented with full vectorization and numerical stability handling
- Extended to multi-class classification using softmax and categorical cross-entropy

### Neural Network (Comparison)
- Simple MLP: 2 hidden layers (64, 32), ReLU activations
- ~4,000 parameters vs 31 for logistic regression
- Trained with mini-batch gradient descent and backpropagation

---

## Evaluation & Results
Models are evaluated using multiple metrics to capture different performance aspects:
- Accuracy
- Log Loss (probability calibration)
- ROC AUC (threshold-independent separability)
- Precision, Recall, F1, and F2 scores (task-sensitive evaluation)

**Key result:**  
Logistic regression achieved ~97% test accuracy and ROC AUC ≈ 0.995, outperforming the neural network on test data. The neural network showed signs of overfitting despite lower training loss.

---

## Key Takeaways
- Simpler models can outperform more complex ones when data is limited and features are informative
- Cross-entropy is an effective training surrogate, but decision thresholds should be tuned post-training
- Model selection should match **data complexity and task requirements**, not theoretical expressiveness
- Evaluation requires צוmultiple metrics - no single score tells the full story

---

## Scope & Limitations
- Uses a train/test split only (no separate validation set)
- Regularization is not explored in depth
- Focused on learning and analysis, not production deployment

---

This repository reflects an end-to-end learning process: from theory to implementation, from metrics to model selection, and from intuition to evidence.
