# Healthcare Cost Prediction

This project builds a regression model to predict healthcare costs based on various features of individuals. The dataset contains information about individuals and their healthcare expenses. The model aims to predict costs accurately within a Mean Absolute Error (MAE) of $3500.

## Dataset
The dataset includes numerical and categorical data about individuals and their healthcare costs. Key steps include:
- Preprocessing categorical data by encoding them as numerical values.
- Splitting the data into training and testing sets (80% training, 20% testing).
- Using the `expenses` column as labels for training and evaluation.

## Objective
The goal is to:
- Train a regression model using 80% of the data (`train_dataset`) and corresponding labels (`train_labels`).
- Evaluate the model on the remaining 20% (`test_dataset` and `test_labels`) to ensure generalization.
- Achieve a Mean Absolute Error (MAE) below $3500 on the test set.

## Implementation Steps

1. **Data Preprocessing**:
   - Handle missing values (if any).
   - Convert categorical columns to numerical using one-hot encoding or label encoding.
   - Normalize or standardize numerical features as needed.

2. **Dataset Splitting**:
   - Split the dataset into `train_dataset` and `test_dataset` (80%-20% split).
   - Separate the `expenses` column to create `train_labels` and `test_labels`.

3. **Model Training**:
   - Use regression algorithms such as Linear Regression, Random Forest, or Gradient Boosting.
   - Train the model on `train_dataset` and `train_labels`.

4. **Evaluation**:
   - Evaluate the model using the test set (`test_dataset` and `test_labels`).
   - Ensure the Mean Absolute Error (MAE) is below $3500.

5. **Visualization**:
   - Predict healthcare costs for the test set and plot the results against actual values to visualize performance.

## Example Usage

```python
# Example data preprocessing
categorical_columns = ['sex', 'region', 'smoker']
for col in categorical_columns:
    dataset[col] = encode_categorical(dataset[col])

# Splitting the dataset
train_dataset, test_dataset, train_labels, test_labels = split_dataset(dataset, 'expenses')

# Model training
model = train_regression_model(train_dataset, train_labels)

# Model evaluation
mae = model.evaluate(test_dataset, test_labels)
print(f"Mean Absolute Error: ${mae}")

# Visualizing predictions
plot_predictions(test_labels, model.predict(test_dataset))
```
