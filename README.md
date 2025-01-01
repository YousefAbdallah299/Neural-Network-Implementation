# Neural Network Implementation

A machine learning project for predicting data using a custom-built neural network.

---

## Features
- Implements a neural network from scratch in Python for educational purposes.
- Utilizes advanced scaling and data handling techniques for improved predictions.
- Calculates performance metrics like Mean Absolute Percentage Error (MAPE).
- Modular design with separate classes for data handling, layers, and neurons.

---

## Table of Contents
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
---

## Getting Started

### Prerequisites
Ensure you have the following installed:
- **Python**: Version 3.8+
- **Libraries**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `openpyxl`


### Architecture
Key Components
- DataHandler: Handles data loading, splitting, scaling, and inverse transformation.
- NeuralNetwork: Custom neural network implementation with forward and backward propagation.
- Layer and Neuron: Represents individual layers and neurons in the network.
Workflow
- Load and preprocess data using DataHandler.
- Train the neural network on scaled training data.
- Predict and evaluate results using test data.

### Usage
Run the main script:
  ```
  python main.py
  ```
Output:
- Predicted vs Actual values for test data.
- Mean Absolute Percentage Error (MAPE).

### Technologies Used
- Python: Core programming language.
- NumPy: For numerical computations.
- Pandas: For data manipulation.
- Scikit-learn: For data splitting and scaling.



