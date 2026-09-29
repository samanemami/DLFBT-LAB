# Final Project

The objective of this project is to implement, train, evaluate, and compare four deep learning architectures for a multiclass classification problem:

- Deep Neural Network (DNN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory Network (LSTM)
- Recurrent Neural Network (RNN)

The project should answer a central question:

> How do DNN, CNN, LSTM, and RNN architectures differ in predictive performance, computational efficiency, training behavior, robustness, and model complexity when applied to the same multiclass sequential classification problem?

Your conclusions must be based on your experimental evidence.

# Required Project Structure

Your submission must follow approximately the following structure:

```text
name_surname_name_surname.zip/
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── training.py
│   └── evaluation.py
│
└── lab3_2026.ipynb
```

## `src/models.py`

You must implement the following three classes:

```python
class DNNClassifier:
    ...

class CNNClassifier:
    ...

class LSTMClassifier:
    ...

class RNNClassifier:
    ...
```

You may use either:

- PyTorch

or

- TensorFlow/Keras

Use the same framework for all models.

The three models must be implemented in `models.py`, **not directly inside the notebook**.

---

## `src/training.py`

Contains reusable training functionality.

For example:

- Training loops
- Validation
- Loss tracking
- Accuracy tracking
- Early stopping
- Training time measurement
- Model checkpointing
- Training history

Avoid writing three completely independent copies of the same training procedure.

---

## `src/evaluation.py`

Contains reusable evaluation functions.

For example:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- Inference timing
- Per-class metrics
- Model comparison

## `lab3_2026.ipynb`

This notebook contains your experiments, visualizations, analysis, and discussion.

The notebook must import the models from your package:

```python
from src.models import (
    DNNClassifier,
    CNNClassifier,
    LSTMClassifier,
    RNNClassifier,
)
```

Do not implement the complete models inside the notebook

# Model Comparison

All models must use:

- Exactly the same training dataset
- Exactly the same validation dataset
- Exactly the same test dataset
- The same evaluation metrics

Training settings should be comparable where possible.

When different settings are necessary because of architectural differences, explain why.

The test set must not be used for model selection or hyperparameter tuning.

# Training

Train all three models.

For every model, record at least:

- Training loss per epoch
- Validation loss per epoch
- Training accuracy per epoch
- Validation accuracy per epoch
- Total training time
- Number of completed epochs

You should use a stopping strategy.

# Metrics

For each model, calculate at least:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score

Also calculate per-class:

- Precision
- Recall
- F1-score
- Support

Generate a confusion matrix for every model.

Explain the difference between macro and weighted metrics and why the distinction matters for this dataset.

# Computational Performance

Compare the computational requirements of the three models.

Measure at least:

- Total training time
- Average time per epoch
- Inference time
- Number of trainable parameters

Where possible, you may investigate:

- Samples processed per second
- CPU/GPU memory consumption
- Model file size

Include the hardware used for your experiments.

# Training

For each architecture, visualize:

### Loss

Plot:

```text
Training Loss
Validation Loss
```

against epoch.

### Accuracy

Plot:

```text
Training Accuracy
Validation Accuracy
```

against epoch.

Use these figures to discuss:

- Convergence
- Overfitting
- Underfitting
- Training stability
- Generalization

# Random Seeds

A single training run is not sufficient to determine whether a result is stable.

Run the main experiment using at least **three random seeds**.

For the principal metrics, report:

```text
mean and standard deviation
```

E.g.,

```text
Model    Macro F1
DNN      mean ± std
CNN      mean ± std
LSTM     mean ± std
RNN      mean ± std
```

# Additional Experiment

After establishing your baseline comparison, perform at least **one additional controlled experiment**.

Choose one important model/training factor, for example:

- Hidden dimension
- Number of layers
- Dropout
- Learning rate
- Batch size
- CNN kernel size
- Number of CNN filters
- LSTM hidden size

Change one factor while keeping other important conditions fixed.

# Confusion Matrix Analysis

Produce one confusion matrix for each model.

An in your project report explain:

- Which classes are easiest to classify?
- Which classes are most frequently confused?
- Are the same classes difficult for all architectures?
- Does one architecture handle particular classes differently?
- What characteristics of the observed data could explain these differences?
