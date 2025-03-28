# IMDb Movie Rating Prediction

## Overview

This project predicts movie ratings using the "IMDb Movies India.csv" dataset. The script performs data loading, preprocessing, feature engineering, and trains a Random Forest Regressor model to predict the ratings.

**GitHub Repository:** [https://github.com/Quantumo0o/IMDb-Data-Rating-Prediction](https://github.com/Quantumo0o/IMDb-Data-Rating-Prediction)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Quantumo0o/IMDb-Data-Rating-Prediction.git](https://www.google.com/search?q=https://github.com/Quantumo0o/IMDb-Data-Rating-Prediction.git)
    cd IMDb-Data-Rating-Prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scipy scikit-learn
    ```

## Usage

1.  **Ensure the dataset `IMDb Movies India.csv` is in the same directory as the Python script (e.g., `Untitled3.ipynb`).** You can usually find this dataset on platforms like Kaggle.

2.  **Run the Python script (e.g., in VS Code or Google Colab):**
    * **In VS Code:** Open the `Untitled3.ipynb` file and run the cells sequentially.
    * **In Google Colab:** Click on the "Open in Colab" badge at the top of this README or upload the `Untitled3.ipynb` file to Google Colab and run the cells.

3.  **Observe the output:** The script will print information about each step of the process, including data loading, cleaning, feature engineering, and the evaluation metrics for the trained models. Visualizations (box plots) will also be displayed.

## Process

The script follows these main steps:

1.  **Import Libraries:** Imports necessary Python libraries for data manipulation, visualization, and machine learning.
2.  **Load Data:** Reads the `IMDb Movies India.csv` dataset into a pandas DataFrame.
3.  **Initial Data Exploration:** Displays the first few rows, data information, and the number of missing values.
4.  **Data Cleaning:**
    * Removes rows with missing values in the 'Rating' column (target variable).
    * Fills missing 'Genre' values with 'Unknown'.
    * Extracts numeric duration in minutes from the 'Duration' column and imputes missing values with the median.
    * Removes rows with missing values in 'Actor 1', 'Actor 2', and 'Actor 3'.
    * Extracts numeric year from the 'Year' column and converts 'Votes' to an integer type.
5.  **Outlier Detection and Removal:**
    * Visualizes the distribution of 'Year' and 'Votes' using box plots before outlier removal.
    * Calculates Z-scores for 'Year' and 'Votes'.
    * Identifies and removes rows where the absolute Z-score for either 'Year' or 'Votes' exceeds a threshold of 2.
    * Visualizes the distribution of 'Year' and 'Votes' using box plots after outlier removal.
6.  **Feature Engineering:**
    * Separates the target variable ('Rating') from the features.
    * Performs one-hot encoding on categorical columns ('Name', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3').
    * Scales numerical columns ('Year', 'Duration', 'Votes') using MinMaxScaler.
    * Concatenates the scaled numerical features and the one-hot encoded categorical features into a final feature matrix.
    * Scales the target variable 'Rating' using MinMaxScaler.
7.  **Model Training and Evaluation:**
    * Splits the data into training and testing sets (70% train, 30% test).
    * **Linear Regression:** Trains a Linear Regression model and evaluates its performance using Mean Squared Error and R-squared score on both the training and test sets.
    * **Tuned Random Forest Regressor:** Trains a Random Forest Regressor model with pre-defined hyperparameters and evaluates its performance using Mean Squared Error and R-squared score on both the training and test sets.

## Dataset

The project uses the "IMDb Movies India.csv" dataset, which contains information about movies listed on IMDb India, including title, year, genre, director, actors, duration, votes, and rating.

## Key Features

* Loads and preprocesses the IMDb movie dataset.
* Handles missing values and performs data type conversions.
* Detects and removes outliers from numerical features.
* Performs one-hot encoding for categorical features.
* Scales numerical features for better model performance.
* Trains and evaluates a Linear Regression model.
* Trains and evaluates a Tuned Random Forest Regressor model for movie rating prediction.
* Provides evaluation metrics (MSE and R-squared) on both training and testing data.
* Includes visualizations (box plots) for outlier detection.

## Model Used

* **Linear Regression:** A simple linear model for baseline comparison.
* **Tuned Random Forest Regressor:** An ensemble learning method known for its good performance on various regression tasks. The hyperparameters used are:
    * `max_depth=None`
    * `max_features='sqrt'`
    * `min_samples_leaf=1`
    * `min_samples_split=3`
    * `n_estimators=50`
    * `random_state=42`

## Evaluation Metrics

The model performance is evaluated using the following metrics:

* **Mean Squared Error (MSE):** The average of the squared differences between the predicted and actual values. Lower values indicate better performance.
* **R-squared (R²):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A value closer to 1 indicates a better fit.

## Potential Improvements

* **More Extensive Exploratory Data Analysis (EDA):** Conduct deeper analysis to understand feature relationships and distributions better.
* **Advanced Feature Engineering:** Explore creating new features from existing ones (e.g., combining actor information, analyzing movie titles).
* **Hyperparameter Tuning:** Implement techniques like GridSearchCV or RandomizedSearchCV to find the optimal hyperparameters for the Random Forest model.
* **Experiment with Other Models:** Try other regression algorithms such as Gradient Boosting Regressors, Support Vector Machines, or Neural Networks.
* **Handling High-Cardinality Categorical Features:** Investigate alternative methods for encoding high-cardinality features like movie names (e.g., target encoding).
* **Cross-Validation:** Implement cross-validation techniques for more robust model evaluation.
* **Save the Trained Model:** Save the trained Random Forest model for future use without retraining.

## License

[Specify your license here, e.g., MIT License]

## Author

[Quantumo0o](https://github.com/Quantumo0o)
