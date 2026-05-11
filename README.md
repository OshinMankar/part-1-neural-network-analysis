# Customer Churn Prediction using Neural Networks

## What is this project?
This project is part of my machine learning assignment where I built a 
neural network from scratch to predict whether a telecom customer will 
churn (leave) or stay. I explored the dataset, preprocessed it, trained 
the model, and ran multiple experiments to understand how different 
settings affect model performance.

## Dataset
The dataset contains information about 2000 telecom customers with 
features like tenure, monthly charges, contract type, payment method, 
and satisfaction score. The goal is to predict the churn column:
- 0 = customer stayed
- 1 = customer left

One big challenge I found was that the dataset is heavily imbalanced —
only 31 out of 2000 customers actually churned (1.55%). This made 
the problem much more interesting to solve.

## Project Structure
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── class_distribution.png
    ├── model_comparison_table.csv
    ├── model_comparison_table.png
    └── evaluation_outputs.png

## How to Run This
1. Clone or download this repository
2. Install all required libraries by running:
   pip install -r requirements.txt
3. Open notebook.ipynb in VS Code or Jupyter Notebook
4. Run all cells from top to bottom

## What I Did (Tasks)
**Task 1 — Explored the data**
Checked the shape, data types, missing values, and visualized 
how many customers churned vs stayed.

**Task 2 — Prepared the data**
Removed the customer ID column, encoded text columns into numbers 
using one-hot encoding, scaled numerical features using StandardScaler, 
and split data into 80% training and 20% testing.

**Task 3 — Built the neural network**
Created a feed-forward neural network with:
- Input layer (24 features)
- Hidden Layer 1 (64 neurons, ReLU)
- Hidden Layer 2 (32 neurons, ReLU)
- Output Layer (1 neuron, Sigmoid)

**Task 4 — Trained and evaluated the model**
Trained for 50 epochs with class weights to handle the imbalance.
Used F1 Score as the main metric instead of accuracy because 
accuracy is misleading when data is this imbalanced.

**Task 5 — Ran 5 experiments**
Changed one setting at a time to see how it affects performance:

| Experiment | What I Changed | F1 Score |
|---|---|---|
| Exp1 | Baseline (64-32, ReLU, lr=0.001) | 0.0000 |
| Exp2 | Deeper network (128-64-32) | 0.2500 |
| Exp3 | Higher learning rate (0.01) | 0.1026 |
| Exp4 | Lower learning rate (0.0001) | 0.2162 |
| Exp5 | Tanh activation, batch size 64 | 0.2500 |

**Task 6 — Wrote my reflections**
Answered questions about weights, biases, activation functions, 
learning rate effects, and whether the model overfit or underfit.

## What I Learned
- Class imbalance is a real challenge — accuracy alone means nothing here
- A deeper network doesn't always mean better results
- Learning rate has a huge impact — too high causes instability, 
  too low means the model never converges properly
- Tanh activation worked surprisingly well for this imbalanced problem
- Always use F1 Score and Recall when dealing with rare events like churn

## Libraries Used
- TensorFlow 2.21.0
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
