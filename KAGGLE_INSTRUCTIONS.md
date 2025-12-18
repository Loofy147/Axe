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

## Second Run with Tuned Hyperparameters

The initial run of this script yielded a low axis accuracy of 51.67%. This second run uses tuned hyperparameters to improve the model's performance. The following changes have been made:

*   `d_model` has been increased from 256 to 512.
*   `num_epochs` has been increased from 20 to 30.
*   The axis loss weight has been increased from 1.0 to 2.0.

These changes are intended to increase the model's capacity, give it more time to learn, and encourage it to focus more on the axis classification task.

## Expected Output

You should see the training progress printed in the notebook's output, including the axis accuracy for each epoch. After the training is complete, the evaluation tests will run, and you will see the model's performance on the test prompts. The expected axis accuracy for this second run is higher than the initial 51.67%.
