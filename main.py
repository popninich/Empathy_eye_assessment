import numpy as np
import pandas as pd
import os

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import tree

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT_FOLDER = "."
SEED = 21

FEATURE_COLS = [
    "avg_change_pupil_left",
    "avg_change_pupil_right",
    "avg_change_total_pupil",
    "peak_pupil_left",
    "peak_pupil_right",
    "peak_total_pupil",
    "short_fixation_count",
    "medium_fixation_count",
    "long_fixation_count",
    "avg_fixation_duration",
    "peak_fixation_duration",
    "saccadic_count",
    "avg_saccadic_durations",
    "avg_saccadic_velocity",
    "peak_saccadic_velocity",
]

# The feature selection by correlation is already prepared.
# You can see more details in the Jupyter Notebook file.
FEATURES_FROM_CORR = [
    "avg_change_pupil_left",
    "avg_change_pupil_right",
    "peak_total_pupil",
    "short_fixation_count",
    "medium_fixation_count",
    "long_fixation_count",
    "avg_fixation_duration",
    "peak_fixation_duration",
    "saccadic_count",
    "avg_saccadic_durations",
    "avg_saccadic_velocity",
    "peak_saccadic_velocity"
]

TARGET_COL = "qcae_score_after"

def read_dataset():
    # I did the train-test split for 80:20 ratio. 
    # The idea of spliting strategy is that for the same participant 
    # experiment, I kept the last 20% of his activities as the test set, 
    # and the remaining first 80% as the training set. So that 
    # I had 60 samples for both training and test sets with different 
    # activity points.
    df = pd.read_csv(os.path.join(ROOT_FOLDER, "data", "data_df.csv"))
    test_df = pd.read_csv(os.path.join(ROOT_FOLDER, "data", "test_data_df.csv"))
    return df, test_df

def preprecess_data(df):
    # Split a dataframe into X and y
    X_train_df = df[FEATURE_COLS]
    y_train_df = df[TARGET_COL]
    
    # The linear regression requires normalization as well in order 
    # to avoid the vanishing gradient problem during the training phase.
    # I used a minmax scalar to transform numbers into [0, 1] range.
    scaler = MinMaxScaler()
    minmax_X_train_df = scaler.fit_transform(X_train_df)
    minmax_X_train_df = pd.DataFrame(minmax_X_train_df, columns=FEATURE_COLS)
    
    # Transform a dataframe to be a numpy array before training
    X_train = minmax_X_train_df.to_numpy()
    y_train = y_train_df.to_numpy()
    return X_train, y_train, scaler

# Feature selection
def _get_X_features_from_corr(X):
    # Features with high correlation are more linearly dependent and 
    # hence have almost the same effect on the dependent variable. 
    # So, when two features have high correlation, we can drop one of 
    # the two features. I used a threshold of 0.9, thus two features should 
    # have more than 0.9 correlation score to identify these two are 
    # highly dependent.
    X_df = pd.DataFrame(X, columns=FEATURE_COLS)
    return X_df[FEATURES_FROM_CORR].to_numpy()

def _compute_sequential_feature_selection(X, y, target_model, method="forward"):
    # A sequential feature selector adds (forward selection) or 
    # removes (backward selection) features to form a feature subset 
    # in a greedy fashion.
    
    if method not in ["forward", "backward"]:
        raise ValueError(f"The `method` argument invalid, it should be only `forward` or `backword`.")
    
    # The cross-validation will use MAE as its metric.
    mae_scores = []
    
    # Loop to find the optimal number of features that should be used
    # The 5-fold cross-validation will be applied to find each loop's MAE.
    for feature_length in range(1, len(FEATURE_COLS)):
        sfs = SequentialFeatureSelector(
            target_model, 
            n_features_to_select=feature_length,
            direction=method,
            scoring="neg_mean_absolute_error",
            cv=5
        )
        # Fit the feature selection model with the provided X and y.
        sfs.fit(X, y)
        # Create a new X data by feature selection transforming.
        X_sfs = sfs.transform(X)
        
        # Collect the mean MAE of the cross-validation as the metric to be compared.
        scores = cross_val_score(
            target_model, 
            X_sfs, 
            y, 
            cv=5,
            scoring="neg_mean_absolute_error"
        )
        mae_scores.append(np.mean(scores))
    
    # Find the best-performance feature size to be the optimal number
    # Using argmax because we used `neg_root_mean_squared_error`, and it returns negative version of an MAE.
    # So it is conforming to the cross validation convention that scorers return higher values for better models.
    optimal_feature_size = np.argmax(mae_scores)+1
    print(f"Feature selection – found the optimal feature size: {optimal_feature_size}, with MAE {-mae_scores[optimal_feature_size-1]:.4f}")

    # Compute the best feature selection process
    best_sfs = SequentialFeatureSelector(
        target_model, 
        n_features_to_select=optimal_feature_size,
        direction=method,
        scoring="neg_mean_absolute_error",
        cv=5
    )
    best_sfs.fit(X, y)
    X_best_sfs = best_sfs.transform(X)
    return best_sfs, X_best_sfs

# Modeling
# Due to the requirements that the models should be explainable, 
# I should to use some simple ML regressor model in this experiment – 
# a `linear regression`, `ridge regression`, and `decision tree regressor`. 
# Because they all can be explained how they produce the prediction via 
# feature importances.
def train_linear_regressions(X_train, y_train):
    print("-"*50)
    print("Start training LINEAR regression models")
    print("-"*50)
    # Train the linear regression model with all features.
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    scores = cross_val_score(
        linear_model, 
        X_train, 
        y_train, 
        cv=5,
        scoring="neg_mean_absolute_error"
    )
    print(f"Linear regression (full features), average MAE: {-np.mean(scores):.4f}")
    print()
    print(f"A prediction equation: y =")
    for idx, feature in enumerate(FEATURE_COLS):
        print(f"\t{linear_model.coef_[idx]:.4f} * {feature}")
    print(f"\t+{linear_model.intercept_:.4f}")
    
    print("-"*50)
    # Train the linear regression model with feature selection by correlation.
    X_train_from_corr = _get_X_features_from_corr(X_train)
    linear_corr_model = LinearRegression()
    linear_corr_model.fit(X_train_from_corr, y_train)
    scores = cross_val_score(
        linear_corr_model, 
        X_train_from_corr, 
        y_train, 
        cv=5,
        scoring="neg_mean_absolute_error"
    )
    print(f"Linear regression (feature selection by corr), average MAE: {-np.mean(scores):.4f}")
    print()
    print(f"A prediction equation: y =")
    for idx, feature in enumerate(FEATURES_FROM_CORR):
        print(f"\t{linear_corr_model.coef_[idx]:.4f} * {feature}")
    print(f"\t+{linear_corr_model.intercept_:.4f}")
    
    print("-"*50)
    # Train the linear regression model with feature selection by SFS.
    # Get the optimal X_train and sfs model from the sequential feature selection
    linear_sfs_model = LinearRegression()
    sfs, X_train_sfs = _compute_sequential_feature_selection(X_train, y_train, linear_sfs_model)

    # The `get_support()` will return a flag to define either that feature should be used or not.
    linear_sfs_features = [FEATURE_COLS[idx] for idx, flag in enumerate(sfs.get_support()) if flag]
    print(f"The optimal features from SFS:")
    for feature in linear_sfs_features:
        print(f"\t- {feature}")
    print()

    # Train a linear regression on these best sfs features
    linear_sfs_model.fit(X_train_sfs, y_train)
    scores = cross_val_score(
        linear_sfs_model, 
        X_train_sfs, 
        y_train, 
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    print(f"Linear regression (feature selection by SFS), average MAE: {-np.mean(scores):.4f}")
    print()
    print(f"A prediction equation: y =")
    for idx, feature in enumerate(linear_sfs_features):
        print(f"\t{linear_sfs_model.coef_[idx]:.4f} * {feature}")
    print(f"\t+{linear_sfs_model.intercept_:.4f}")
    return linear_model, linear_corr_model, linear_sfs_model, linear_sfs_features
    
def train_ridge_regressions(X_train, y_train):
    print("-"*50)
    print("Start training RIDGE regression models")
    print("-"*50)
    
    # Hyperparams
    ridge_params = {
        "alpha": np.arange(0, 50, 0.5)
    }
    
    # Train a ridge model with full features
    # Fine-tune the best hyperparameters using a grid search, and get the best-performance model.
    ridge_grid_cv = GridSearchCV(
        Ridge(random_state=SEED), 
        param_grid=ridge_params,
        scoring="neg_mean_absolute_error",
        cv=5,
        refit=True
    )
    ridge_grid_cv.fit(X_train, y_train)
    # Select the best estimator
    ridge_model = ridge_grid_cv.best_estimator_
    print(f"Ridge regression (full features), best params: {ridge_grid_cv.best_params_}, average MAE: {-ridge_grid_cv.best_score_:.4f}")
    print()
    print(f"A prediction equation: y =")
    for idx, feature in enumerate(FEATURE_COLS):
        print(f"\t{ridge_model.coef_[idx]:.4f} * {feature}")
    print(f"\t+{ridge_model.intercept_:.4f}")
    
    print("-"*50)
    # Train a ridge model with feature selection by correlation
    # Get X_train from the feature selection by correlation function
    X_train_from_corr = _get_X_features_from_corr(X_train)

    # Fine-tune the best hyperparameters using a grid search, and get the best-performance model.
    ridge_grid_corr_cv = GridSearchCV(
        Ridge(random_state=SEED), 
        param_grid=ridge_params,
        scoring="neg_mean_absolute_error",
        cv=5,
        refit=True
    )
    ridge_grid_corr_cv.fit(X_train_from_corr, y_train)

    # Select the best estimator
    ridge_corr_model = ridge_grid_corr_cv.best_estimator_

    print(f"Ridge regression (feature selection by corr), best params: {ridge_grid_corr_cv.best_params_}, average MAE: {-ridge_grid_corr_cv.best_score_:.4f}")
    print()
    print(f"A prediction equation: y =")
    for idx, feature in enumerate(FEATURES_FROM_CORR):
        print(f"\t{ridge_corr_model.coef_[idx]:.4f} * {feature}")
    print(f"\t+{ridge_corr_model.intercept_:.4f}")
    
    print("-"*50)
    # Train a ridge model with feature selection by SFS
    # Get the optimal X_train and sfs model from the sequential feature selection
    ridge_base_model = Ridge(random_state=SEED)
    sfs, X_train_sfs = _compute_sequential_feature_selection(X_train, y_train, ridge_base_model)

    # The `get_support()` will return a flag to define either that feature should be used or not.
    ridge_sfs_features = [FEATURE_COLS[idx] for idx, flag in enumerate(sfs.get_support()) if flag]
    print(f"The optimal features from SFS:")
    for feature in ridge_sfs_features:
        print(f"\t- {feature}")
    print()

    # Fine-tune the best hyperparameters using a grid search, and get the best-performance model.
    ridge_grid_sfs_cv = GridSearchCV(
        ridge_base_model, 
        param_grid=ridge_params,
        scoring="neg_mean_absolute_error",
        cv=5,
        refit=True
    )
    ridge_grid_sfs_cv.fit(X_train_sfs, y_train)

    # Select the best estimator
    ridge_sfs_model = ridge_grid_sfs_cv.best_estimator_

    print(f"Ridge regression (feature selection by SFS), best params: {ridge_grid_sfs_cv.best_params_}, average MAE: {-ridge_grid_sfs_cv.best_score_:.4f}")
    print()
    print(f"A prediction equation: y =")
    for idx, feature in enumerate(ridge_sfs_features):
        print(f"\t{ridge_sfs_model.coef_[idx]:.4f} * {feature}")
    print(f"\t+{ridge_sfs_model.intercept_:.4f}")
    
    return ridge_model, ridge_corr_model, ridge_sfs_model, ridge_sfs_features
    
def train_decision_tree_regressions(X_train, y_train):
    print("-"*50)
    print("Start training DECISION TREE regression models")
    print("-"*50)
    
    # Hyperparams
    tree_params = {
        "splitter": ["best", "random"],
        "max_depth": np.arange(3, 12),
        "min_samples_leaf": np.arange(2, 20, 2)
    }
    
    # Train a decision tree with full features
    # Fine-tune the best hyperparameters using a grid search, and get the best-performance model
    tree_grid_cv = GridSearchCV(
        DecisionTreeRegressor(random_state=SEED), 
        param_grid=tree_params,
        scoring="neg_mean_absolute_error",
        cv=5,
        refit=True
    )
    tree_grid_cv.fit(X_train, y_train)

    # Select the best estimator
    tree_model = tree_grid_cv.best_estimator_

    print(f"Decision tree regressor (full features), best params: {tree_grid_cv.best_params_}, MAE: {-tree_grid_cv.best_score_:.4f}")
    print()

    print(f"Feature importances:")
    for idx, feature in enumerate(FEATURE_COLS):
        print(f"\t{feature}: {tree_model.feature_importances_[idx]:.4f}")
    print()

    # Save the tree diagram
    plt.figure()
    tree.plot_tree(tree_model, filled=True, feature_names=FEATURE_COLS)  
    plt.savefig("src/tree-full_features.pdf")
    
    print("-"*50)
    # Train a decision tree with full selection by correlations
    # Get X_train from the feature selection by correlation function
    X_train_from_corr = _get_X_features_from_corr(X_train)

    # Fine-tune the best hyperparameters using a grid search, and get the best-performance model
    tree_grid_corr_cv = GridSearchCV(
        DecisionTreeRegressor(random_state=SEED), 
        param_grid=tree_params,
        scoring="neg_mean_absolute_error",
        cv=5,
        refit=True
    )
    tree_grid_corr_cv.fit(X_train_from_corr, y_train)

    # Select the best estimator
    tree_corr_model = tree_grid_corr_cv.best_estimator_

    print(f"Decision tree regressor (features from corr), best params: {tree_grid_corr_cv.best_params_}, average MAE: {-tree_grid_corr_cv.best_score_:.4f}")
    print()

    print(f"Feature importances:")
    for idx, feature in enumerate(FEATURES_FROM_CORR):
        print(f"\t{feature}: {tree_corr_model.feature_importances_[idx]:.4f}")
    print()

    # Save the tree diagram
    plt.figure()
    tree.plot_tree(tree_corr_model, filled=True, feature_names=FEATURES_FROM_CORR)  
    plt.savefig("src/tree-features_from_corr.pdf")
    
    print("-"*50)
    # Train a decision tree with feature selection by SFS
    # Get the optimal X_train and sfs model from the sequential feature selection
    tree_base_model = DecisionTreeRegressor(random_state=SEED)
    sfs, X_train_sfs = _compute_sequential_feature_selection(X_train, y_train, tree_base_model)

    # The `get_support()` will return a flag to define either that feature should be used or not.
    tree_sfs_features = [FEATURE_COLS[idx] for idx, flag in enumerate(sfs.get_support()) if flag]
    print(f"The optimal features from SFS:")
    for feature in tree_sfs_features:
        print(f"\t- {feature}")
    print()

    # Fine-tune the best hyperparameters using a grid search, and get the best-performance model.
    tree_grid_sfs_cv = GridSearchCV(
        tree_base_model, 
        param_grid=tree_params,
        scoring="neg_mean_absolute_error",
        cv=5,
        refit=True
    )
    tree_grid_sfs_cv.fit(X_train_sfs, y_train)

    # Select the best estimator
    tree_sfs_model = tree_grid_sfs_cv.best_estimator_

    print(f"Decision tree regressor (features from SFS), best params: {tree_grid_sfs_cv.best_params_}, average MAE: {-tree_grid_sfs_cv.best_score_:.4f}")
    print()

    print(f"Feature importances:")
    for idx, feature in enumerate(tree_sfs_features):
        print(f"\t{feature}: {tree_model.feature_importances_[idx]:.4f}")
    print()

    # Save the tree diagram
    plt.figure()
    tree.plot_tree(tree_sfs_model, filled=True, feature_names=tree_sfs_features)  
    plt.savefig("src/tree-features_sfs.pdf")
    
    return tree_model, tree_corr_model, tree_sfs_model, tree_sfs_features


if __name__ == '__main__':
    if not os.path.exists(os.path.join(ROOT_FOLDER, "src")):
        os.makedirs(os.path.join(ROOT_FOLDER, "src"))
        
    df, test_df = read_dataset()
    X_train, y_train, normalizer = preprecess_data(df)
    
    linear_model, linear_corr_model, linear_sfs_model, linear_sfs_features = train_linear_regressions(X_train, y_train)
    ridge_model, ridge_corr_model, ridge_sfs_model, ridge_sfs_features = train_ridge_regressions(X_train, y_train)
    tree_model, tree_corr_model, tree_sfs_model, tree_sfs_features = train_decision_tree_regressions(X_train, y_train)
    
    # Evaluate the models
    X_test_df = test_df[FEATURE_COLS]
    y_test_df = test_df[TARGET_COL]
    
    # The normalization should be only applying to the test set, not to fit again
    minmax_X_test_df = normalizer.transform(X_test_df)
    minmax_X_test_df = pd.DataFrame(minmax_X_test_df, columns=FEATURE_COLS)
    
    # Prepare X_test for each model because it has the different set of features
    X_test_linear = minmax_X_test_df[FEATURE_COLS].to_numpy()
    X_test_linear_corr = minmax_X_test_df[FEATURES_FROM_CORR].to_numpy()
    X_test_linear_sfs = minmax_X_test_df[linear_sfs_features].to_numpy()

    X_test_ridge = minmax_X_test_df[FEATURE_COLS].to_numpy()
    X_test_ridge_corr = minmax_X_test_df[FEATURES_FROM_CORR].to_numpy()
    X_test_ridge_sfs = minmax_X_test_df[ridge_sfs_features].to_numpy()

    X_test_tree = minmax_X_test_df[FEATURE_COLS].to_numpy()
    X_test_tree_corr = minmax_X_test_df[FEATURES_FROM_CORR].to_numpy()
    X_test_tree_sfs = minmax_X_test_df[tree_sfs_features].to_numpy()
    
    # Prepare y to be a numpy array
    y_test = y_test_df.to_numpy()
    
    # Compute predictions for all considering models
    y_pred_linear = linear_model.predict(X_test_linear)
    y_pred_linear_corr = linear_corr_model.predict(X_test_linear_corr)
    y_pred_linear_sfs = linear_sfs_model.predict(X_test_linear_sfs)

    y_pred_ridge = ridge_model.predict(X_test_ridge)
    y_pred_ridge_corr = ridge_corr_model.predict(X_test_ridge_corr)
    y_pred_ridge_sfs = ridge_sfs_model.predict(X_test_ridge_sfs)

    y_pred_tree = tree_model.predict(X_test_tree)
    y_pred_tree_corr = tree_corr_model.predict(X_test_tree_corr)
    y_pred_tree_sfs = tree_sfs_model.predict(X_test_tree_sfs)
    
    # Prepare errors for all models in term of actual error (ME) and absolute error (MAE)
    me_linear = []
    me_linear_corr = []
    me_linear_sfs = []
    me_ridge = []
    me_ridge_corr = []
    me_ridge_sfs = []
    me_tree = []
    me_tree_corr = []
    me_tree_sfs = []

    mae_linear = []
    mae_linear_corr = []
    mae_linear_sfs = []
    mae_ridge = []
    mae_ridge_corr = []
    mae_ridge_sfs = []
    mae_tree = []
    mae_tree_corr = []
    mae_tree_sfs = []

    for idx, y in enumerate(y_test):
        me_linear.append(y_pred_linear[idx]-y)
        me_linear_corr.append(y_pred_linear_corr[idx]-y)
        me_linear_sfs.append(y_pred_linear_sfs[idx]-y)
        me_ridge.append(y_pred_ridge[idx]-y)
        me_ridge_corr.append(y_pred_ridge_corr[idx]-y)
        me_ridge_sfs.append(y_pred_ridge_sfs[idx]-y)
        me_tree.append(y_pred_tree[idx]-y)
        me_tree_corr.append(y_pred_tree_corr[idx]-y)
        me_tree_sfs.append(y_pred_tree_sfs[idx]-y)
        
        mae_linear.append(np.abs(y_pred_linear[idx]-y))
        mae_linear_corr.append(np.abs(y_pred_linear_corr[idx]-y))
        mae_linear_sfs.append(np.abs(y_pred_linear_sfs[idx]-y))
        mae_ridge.append(np.abs(y_pred_ridge[idx]-y))
        mae_ridge_corr.append(np.abs(y_pred_ridge_corr[idx]-y))
        mae_ridge_sfs.append(np.abs(y_pred_ridge_sfs[idx]-y))
        mae_tree.append(np.abs(y_pred_tree[idx]-y))
        mae_tree_corr.append(np.abs(y_pred_tree_corr[idx]-y))
        mae_tree_sfs.append(np.abs(y_pred_tree_sfs[idx]-y))
        
    # Prepare dataframes to be plotted
    plot_me_df = pd.DataFrame.from_dict({
        "ME": me_linear + me_linear_corr + me_linear_sfs + me_ridge + me_ridge_corr + me_ridge_sfs + me_tree + me_tree_corr + me_tree_sfs,
        "model": ["Linear (all features)"]*len(y_test)\
                + ["Linear (features from corr)"]*len(y_test) \
                + ["Linear (features from SFS)"]*len(y_test) \
                + ["Ridge (all features)"]*len(y_test)\
                + ["Ridge (features from corr)"]*len(y_test) \
                + ["Ridge (features from SFS)"]*len(y_test) \
                + ["Decision tree (all features)"]*len(y_test)\
                + ["Decision tree (features from corr)"]*len(y_test) \
                + ["Decision tree (features from SFS)"]*len(y_test)
    })

    plot_mae_df = pd.DataFrame.from_dict({
        "MAE": mae_linear + mae_linear_corr + mae_linear_sfs + mae_ridge + mae_ridge_corr + mae_ridge_sfs + mae_tree + mae_tree_corr + mae_tree_sfs,
        "model": ["Linear (all features)"]*len(y_test)\
                + ["Linear (features from corr)"]*len(y_test) \
                + ["Linear (features from SFS)"]*len(y_test) \
                + ["Ridge (all features)"]*len(y_test)\
                + ["Ridge (features from corr)"]*len(y_test) \
                + ["Ridge (features from SFS)"]*len(y_test) \
                + ["Decision tree (all features)"]*len(y_test)\
                + ["Decision tree (features from corr)"]*len(y_test) \
                + ["Decision tree (features from SFS)"]*len(y_test)
    })
    
    # Save plot dataframes
    plot_me_df.to_csv("src/plot_me_df.csv", header=True, index=False)
    plot_mae_df.to_csv("src/plot_mae_df.csv", header=True, index=False)
    
    # Show test scores for each model
    print("-"*50)
    print("Model evaluation on test set:\n")
    print(plot_mae_df.groupby(["model"]).mean())