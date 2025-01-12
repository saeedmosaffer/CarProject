# Car Evaluation Decision Tree Classifier

This project demonstrates the use of the **C4.5 Decision Tree Algorithm** (implemented as **J48** in Weka) to classify cars based on the Car Evaluation Dataset. The dataset includes car attributes like safety, buying cost, maintenance, and others, with four possible classifications: `unacc`, `acc`, `good`, and `vgood`.

## Project Overview

The project focuses on:
- **Class Distribution Analysis**: Understanding the dataset's target class distribution.
- **Model Training and Evaluation**:
  - **M1**: Trained on 70% of the data, tested on 30%.
  - **M2**: Trained on 50% of the data, tested on 50%.
- **Performance Comparison**: Comparing the accuracy and F1-scores of the two models.
- **Decision Tree Visualization**: Generating and visualizing the decision trees for both models.

## Key Results

| Model | Training/Test Split | Accuracy (%) | F1-Score |
|-------|----------------------|--------------|----------|
| M1    | 70% / 30%           | 90.93        | 0.9104   |
| M2    | 50% / 50%           | 85.53        | 0.8585   |

- M1 outperformed M2 due to its larger training dataset.

## Features

- **Dataset Preprocessing**: Shuffling and splitting the data to avoid bias.
- **Evaluation Metrics**: Accuracy, F1-score, confusion matrices, and detailed accuracy by class.
- **Visualization**: Decision trees generated in DOT format, viewable using GraphViz.

- **Report**:[Project 2 Report.pdf](https://github.com/user-attachments/files/18391331/Project.2.Report.pdf)

