# CS-589-Final-Project

# 🧠 COMPSCI 589 Final Project – Spring 2025

This repository contains original implementations of fundamental machine learning algorithms and their application on multiple real-world datasets as part of the final project for COMPSCI 589 at UMass Amherst.

## 📌 Project Objectives

- Implement **Decision Trees**, **Random Forests**, and **Naive Bayes** from scratch without ML libraries like scikit-learn or PyTorch.
- Evaluate and compare their performance on diverse datasets through extensive experimentation.
- Explore **hyperparameter tuning**, **cross-validation**, and **model robustness**.
- Earned **extra credit** by including a challenging **multiclass, mixed-type dataset** for fertilizer recommendation.

## 🗂️ Datasets Used

| Dataset                        | Task Type          | Classes | Features     | Notes                                |
|-------------------------------|--------------------|---------|--------------|--------------------------------------|
| Handwritten Digits (Sklearn)  | Image classification| 10      | 64 (numeric) | Pixel-intensity classification       |
| Parkinson’s Detection         | Binary classification| 2     | 22 (numeric) | Voice measurement for diagnosis      |
| Rice Grain Types              | Binary classification| 2     | 7 (numeric)  | Morphological kernel features        |
| Credit Approval               | Binary classification| 2     | 15 (mixed)   | Contains categorical + numerical     |
| Fertilizer Recommendation     | Multiclass classification| 10 | Mixed       | Kaggle dataset, extra credit task    |

## 🔍 Algorithms Implemented

- ✅ **Decision Trees**  
  Custom Gini/Entropy-based split logic, depth-tuning, suitable for mixed data.

- 🌲 **Random Forests**  
  Custom bagging ensemble of trees with tunable estimators.

- 🧮 **Naive Bayes**  
  Applied to multiclass fertilizer task; Laplace smoothing (α) tuning over wide ranges.

## ⚙️ Methodology

- Implemented using only **NumPy**, **Matplotlib**, and **Python's core libraries**.
- Evaluated using **10-fold Stratified Cross-Validation**.
- Plotted **learning curves** and **performance graphs** for hyperparameter sensitivity.

## 📈 Sample Results

| Dataset           | Best Model         | Accuracy | F1 Score |
|------------------|--------------------|----------|----------|
| Digits           | Random Forest (5 trees) | 0.881    | 0.880    |
| Parkinson’s       | Decision Tree (depth 4) | 0.877    | 0.815    |
| Rice              | Decision Tree (depth 4) | 0.921    | 0.919    |
| Credit Approval   | Random Forest (20 trees)| 0.815    | 0.812    |
| Fertilizer (Extra)| Decision Tree (depth 12)| 0.986    | 0.949    |

## 📁 Repository Structure

├── code/ # Python files with all algorithm implementations
├── data/ # Cleaned datasets used in the project
├── results/ # Output plots and metrics
├── Final_Report.pdf # Complete project report (with graphs, tables, and analysis)
└── README.md # Project overview (this file)


## 🏆 Extra Credit

- ✅ Evaluated all four datasets using **three algorithms** (DT, RF, NB)
- ✅ Included a **custom multiclass dataset** with mixed data types
- ✅ Achieved 100% F1 score for small-α values using Naive Bayes

## 📧 Author

**Kartikay Gupta**  
Graduate Student – MSCS  
📨 kartikaygupt@umass.edu  
🔗 [LinkedIn](https://linkedin.com/in/kartikay77)

---

> _Note: All implementations are original and adhere to the no-ML-library rule defined in the project instructions._
