# 🧠 COMPSCI 589 Final Project – Spring 2025

This repository contains custom implementations of core machine learning algorithms and their application across a variety of datasets. Developed as the final project for COMPSCI 589 at UMass Amherst.

> 👥 **Team Members**  
> • Kartikay Gupta – Implemented Decision Tree, Random Forest, Naive Bayes  
> • Jeffrey Deng – Implemented Neural Network (MLP with backpropagation)

---

## 🎯 Project Goals

- Apply and compare original implementations of key ML algorithms
- Analyze algorithm behavior using **10-fold stratified cross-validation**
- Explore hyperparameter tuning and visualize results via matplotlib
- **No use of high-level ML libraries** like scikit-learn for modeling

---

## 📊 Datasets Used

| Dataset            | Classes | Features | Type                            |
|--------------------|---------|----------|---------------------------------|
| Digits (Sklearn)   | 10      | 64       | Image classification            |
| Parkinson’s        | 2       | 22       | Biomedical voice data           |
| Rice Grain         | 2       | 7        | Morphological image features    |
| Credit Approval    | 2       | 15       | Mixed categorical + numerical   |
| Fertilizer (Kaggle)| 10      | Mixed    | Soil/crop conditions (Extra)    |

---

## 🧠 Algorithms Implemented

| Algorithm        | Contributor     | Notes                                        |
|------------------|------------------|----------------------------------------------|
| Decision Tree    | Kartikay Gupta   | From-scratch, entropy/Gini split, label encoded |
| Random Forest    | Kartikay Gupta   | Ensemble with bootstrap and aggregation      |
| Naive Bayes      | Kartikay Gupta   | Multiclass support, tuned smoothing factor α |
| Neural Network   | Jeffrey Deng     | Multilayer Perceptron with backpropagation   |

---

## 📈 Performance Summary

| Algorithm        | Digits (Acc/F1) | Parkinson (Acc/F1) | Rice (Acc/F1) | Credit (Acc/F1) | Fertilizer (Acc/F1) |
|------------------|------------------|----------------------|----------------|------------------|----------------------|
| Decision Tree    | 0.8202 / 0.8198 | 0.8770 / 0.8152     | 0.9213 / 0.9192| 0.8133 / 0.8100 | 0.9861 / 0.9493     |
| Random Forest    | 0.8812 / 0.8797 | 0.8253 / 0.7855     | 0.9055 / 0.9033| 0.8149 / 0.8121 | — / —               |
| Naive Bayes      | — / —           | — / —               | — / —         | — / —           | 1.0000 / 1.0000     |
| Neural Network   | 0.9806 / 0.9803 | 0.8974 / 0.8427     | 0.9449 / 0.9436| 0.9160 / 0.9157 | — / —               |

✔️ **Highlights**:  
- Neural Networks outperformed all others in **Digits, Parkinson’s, Rice, and Credit** datasets.  
- Naive Bayes achieved **perfect scores** on the Fertilizer dataset.  
- Decision Trees showed consistently strong and interpretable results across datasets.

---

## 📂 Project Structure

├── code/ # All algorithm implementations (no scikit-learn models used)

--
├── data/ # Datasets in CSV format

--
├── results/ # Output plots, performance tables

--
├── Final_Report.pdf # Official final submission with figures and write-up

--
└── README.md # This file
--

---

## 🏆 Extra Credit

- ✅ Used 3+ algorithms for multiple datasets  
- ✅ Included a **challenging multiclass + mixed-type dataset**  
- ✅ Perfect score (1.000) achieved using Naive Bayes on Fertilizer data

---

## 🧑‍💻 Authors

- **Kartikay Gupta**  
  📧 kartikaygupt@umass.edu  
  🔗 [LinkedIn](https://linkedin.com/in/kartikay77)

- **Jeffrey Deng**  
  📧 jjdeng@umass.edu

---

> ⚠️ Models in this repo are implemented fully from scratch per course policy. Neural Network implementation and performance analysis credited to Jeffrey Deng.


