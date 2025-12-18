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

## Fifth Run: Addressing Underfitting

The fourth training run was stable, but the model's performance was poor, indicating that it was underfitting. To address this, the following changes have been made to the learning rate and scheduler:

*   **Increased Learning Rate:** The learning rate has been increased to `2e-4`.
*   **Adjusted Scheduler:** The `OneCycleLR` scheduler has been adjusted to have a longer warm-up phase (`pct_start=0.3`) and a less aggressive peak.

These changes are intended to help the model learn more effectively from the data, while the warm-up phase should prevent the training instability seen in previous runs.

## Expected Output

You should see the training progress printed in the notebook's output. The training should be stable, and the validation accuracy should be higher than in the previous run. The target axis accuracy for this run is between 75% and 85%.
