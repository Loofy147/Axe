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

## Fourth Run: Stabilizing Training

The third training run failed due to training instability, likely caused by exploding gradients. The model's performance collapsed mid-training, with the loss increasing dramatically and the accuracy dropping to zero. This is a common issue when the learning rate is too high.

To address this, the following change has been made:

*   **Reduced Learning Rate:** The learning rate has been reduced from `1e-4` to `5e-5`. This will make the training updates smaller and less likely to cause the gradients to explode, leading to a more stable learning process.

## Expected Output

You should see the training progress printed in the notebook's output. The training should now be stable, without the sudden collapse in performance seen in the previous run. The target axis accuracy for this run is between 75% and 85%.
