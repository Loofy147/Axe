# Kaggle Training Instructions

This guide will walk you through setting up and running the futures model training and evaluation in a Kaggle notebook.

## Step 1: Create a New Kaggle Notebook

1.  Go to [Kaggle](https://www.kaggle.com/) and click on "Create Notebook" in the left-hand menu.
2.  Choose a "Python" notebook.

## Step 2: Upload the Dataset

1.  In your Kaggle notebook, click on "Add Data" in the top-right corner.
2.  Click on "Upload" and select the `futures_dataset.json` file.
3.  The dataset will be uploaded to `/kaggle/input/futures-dataset/`.

## Step 3: Add the Code

1.  In the first cell of your notebook, add the following code to install the necessary libraries:

    ```python
    !pip install torch transformers
    ```

2.  In the second cell, copy and paste the entire contents of the `kaggle_runner.py` script.

## Step 4: Run the Notebook

1.  Click on "Run All" in the top menu of your notebook.
2.  The notebook will execute the code, which will:
    *   Install the required libraries.
    *   Train the model on the `futures_dataset.json` data.
    *   Save the best model checkpoint to `/kaggle/working/checkpoint.pt`.
    *   Run the evaluation tests to assess the model's performance.

## Third Run: Addressing Overfitting

The second training run achieved 100% axis accuracy, but the generated text was nonsensical. This indicates that the model has overfit the training data. This third run introduces changes to combat overfitting:

*   **Dropout Regularization:** Dropout has been added to the model architecture. This technique randomly deactivates a fraction of neurons during training, which prevents the model from becoming too reliant on any single neuron and encourages it to learn more robust features.
*   **Reduced Model Size:** The model's capacity (`d_model`) has been reverted from 512 back to 256. A smaller model is less prone to overfitting.

These changes are intended to help the model generalize better to new data and learn the underlying semantic meaning of the axes, rather than just memorizing the training set.

## Expected Output

You should see the training progress printed in the notebook's output. The target axis accuracy for this run is between 75% and 85%. The generated text should be more coherent than in the previous run.
