import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from main import X_train, X_test, y_train, y_test
from sklearn.neighbors import KNeighborsClassifier
def train_and_evaluate_with_mlflow(model, param_grid, X_train, X_test, y_train, y_test, model_name, **kwargs):
    """
    Train a machine learning model using GridSearchCV and evaluate its performance,
    with all results and the model itself logged to MLflow.

    Parameters:
    - model: The machine learning model to train.
    - param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
    - X_train: Training data features.
    - X_test: Testing data features.
    - y_train: Training data labels.
    - y_test: Testing data labels.
    - model_name: The name of the model (for MLflow logging).
    - **kwargs: Additional keyword arguments to pass to the GridSearchCV.

    Returns:
    - The best estimator from GridSearchCV.
    """
    with mlflow.start_run():
        mlflow.set_experiment("Student Status Prediction")

        # Perform grid search to find the best parameters
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, **kwargs)
        grid_search.fit(X_train, y_train)

        # Extract information from the grid search for logging
        cv_results_df = pd.DataFrame(grid_search.cv_results_)

        # Get the top 5 best parameter combinations by rank_test_score
        top5_results = cv_results_df.sort_values('rank_test_score').head(5)

        # Log the best parameters
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)

        # Evaluate the model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Log the performance metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("f1", f1_score(y_test, y_pred, average='weighted'))

        # Log the top 5 best results as an artifact
        top5_results.to_csv("top5_results.csv", index=False)
        mlflow.log_artifact("top5_results.csv")

        # Log the best model in MLflow
        mlflow.sklearn.log_model(best_model, model_name)

        return best_model



# Decision Tree hyperparameters
dt_param_grid = {
    'max_depth': [3, 4,5,6, 10],
    'min_samples_leaf': [1, 2, 4]
}

# KNN hyperparameters
k_list = list(range(1, 101))
knn_param_grid = {
    'n_neighbors': k_list
}

# Set the MLflow experiment name
# mlflow.set_experiment("Model Comparison Experiment")

# Run Decision Tree experiment
train_and_evaluate_with_mlflow(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    X_train, X_test, y_train, y_test,
    model_name="DecisionTree",
    cv=5
)
print("hello")
# Run KNN experiment
train_and_evaluate_with_mlflow(
    KNeighborsClassifier(),
    knn_param_grid,
    X_train, X_test, y_train, y_test,
    model_name="KNN",
    cv=5
)
