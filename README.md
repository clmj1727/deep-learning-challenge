# Alphabet Soup Deep Learning Model Report

## Purpose of the Analysis

The purpose of this analysis was to develop a binary classification model using deep learning to predict the success of grant applications to a fictional nonprofit funding organization, **Alphabet Soup**. The dataset contained various features related to the organization’s characteristics, with the goal of identifying patterns that indicate the likelihood of funding approval. The intended outcome was to create a model that could predict outcomes with at least **75% accuracy**.

---

## Results

### Data Preprocessing

#### What variable(s) are the target(s) for your model?

- `IS_SUCCESSFUL`: This binary variable indicates whether the funding application was approved (1) or not (0). It serves as the target variable for the classification task.

#### What variable(s) are the features for your model?

- All variables except `IS_SUCCESSFUL`, `EIN`, and `NAME` were treated as features.
- After binning and encoding, features included:
  - `APPLICATION_TYPE`, `CLASSIFICATION`, `AFFILIATION`, `INCOME_AMT`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `SPECIAL_CONSIDERATIONS`
  - All categorical features were converted using `pd.get_dummies()` for one-hot encoding.

#### What variable(s) should be removed from the input data because they are neither targets nor features?

- `EIN`: This is a unique identifier for organizations and does not contribute to the prediction.
- `NAME`: Organization names are not predictive and were also removed.

---

### Compiling, Training, and Evaluating the Model

#### How many neurons, layers, and activation functions did you select for your neural network model, and why?

The neural network was composed as follows:

- **Input Layer**: Number of input features (determined dynamically)
- **First Hidden Layer**: 80 neurons, ReLU activation
- **Second Hidden Layer**: 30 neurons, ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation (for binary output)

The number of neurons was selected based on a general heuristic (between the input size and the output size) and experimentation to balance performance and training time.

#### Were you able to achieve the target model performance?

- **No**, the model did **not** reach the target accuracy of **75%**. The model's accuracy plateaued below the desired threshold even after tuning.

#### What steps did you take in your attempts to increase model performance?

- Binned infrequent categories in `APPLICATION_TYPE` and `CLASSIFICATION` into an "Other" category to reduce dimensionality.
- Applied one-hot encoding to all categorical variables using `pd.get_dummies()`.
- Scaled numerical features using `StandardScaler` to normalize feature values.
- Experimented with the number of neurons in each layer.
- Adjusted the number of hidden layers and epochs.

---

## Summary of Results

The first deep learning model produced moderate accuracy but **did not meet the 75% accuracy threshold**. The performance was hindered possibly due to:

- High dimensionality of one-hot encoded features
- Limited training data for deep learning models
- Potential overfitting or underfitting based on model architecture

The final optimized neural network model **exceeded the target accuracy of 75%**, achieving **79.13% accuracy** on the test set. This result demonstrates that a well-preprocessed and tuned deep learning model can effectively classify successful vs. unsuccessful applications based on organizational data.

Optimization included:
- Dropping non-benificial categories such as `EIN`, `STATUS`, and `SPECIAL_CONSIDERATIONS`
- Binning infrequent categories in `NAME`, `APPLICATION_TYPE` and `CLASSIFICATION` into an "Other" category to reduce dimensionality (this was one more than the original model)
- Applied one-hot encoding to categorical variables
- Scaled features using StandardScaler
- Tuned architecture (neurons/layers)
- Experimented with training epochs and batch size

Visualizations of improvement in training are shown below for both models:

<p float="left">
  <img src="https://github.com/clmj1727/deep-learning-challenge/blob/main/Model%201%3A%20Training%20Accuracy%20Over%20Epochs.png" width="45%" />
  <img src="https://github.com/clmj1727/deep-learning-challenge/blob/main/Optimized%20Model%3A%20Training%20Accuracy%20Over%20Epochs.png" width="45%" />
</p>


---

## Recommendations: Try a Different Model

For structured, tabular data like this, a **tree-based model** may be more appropriate. Suggested alternatives include:

### Random Forest

- Handles categorical and numerical variables well
- Naturally resists overfitting
- Provides feature importance insights

### XGBoost (Extreme Gradient Boosting)

- Highly efficient and scalable
- Often outperforms deep learning on tabular data
- Includes built-in regularization

These models typically perform better on tabular data because they are designed to handle heterogeneous feature types, missing values, and feature interactions more naturally than deep learning models.

---
# Acknowledgements:

Special thanks to Dr. Carl Arrington for guidance during the Intro to Advanced Machine Learning, Deep Learning, and Neural Networks lectures. Some snippets and logic were developed following in-class tutorial support and discussions.

