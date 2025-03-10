
## Usage

1.  **Clone the repository:**

    ```
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

4.  **Run the main script:**

    ```
    python main.py
    ```

    This will train the model on the IMDB dataset, evaluate its performance, and display the results.

## Code Explanation

The `binary_classification_nlp.py` script performs the following steps:

1.  **Data Loading and Preprocessing:** Loads the IMDB dataset using Keras, limits the vocabulary size, and converts the text data into numerical vectors using one-hot encoding.
2.  **Model Definition:** Defines a sequential model with two dense layers and a sigmoid output layer.
3.  **Model Compilation:** Configures the model with the `rmsprop` optimizer, `binary_crossentropy` loss function, and `accuracy` metric.
4.  **Training and Validation:** Trains the model on a portion of the training data and validates its performance on a separate validation set.  The training history is plotted to visualize the training and validation loss.
5.  **Evaluation:** Evaluates the trained model on the test dataset and prints the loss and accuracy.
6.  **Prediction:** Demonstrates how to use the trained model to predict the sentiment of new reviews.

## Performance

The model typically achieves an accuracy of around 88% on the test dataset after training for a few epochs. The exact performance may vary depending on the training parameters and the random initialization of the model.  Overfitting is a common issue, so monitoring the validation loss and adjusting the number of epochs is crucial.
