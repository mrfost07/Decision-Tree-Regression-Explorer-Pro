import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
import graphviz
import pickle
import json
import time
from typing import Tuple, List, Dict, Any

@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Load and cache the uploaded CSV file."""
    df = pd.read_csv(file)
    
    # Handle unnamed columns
    unnamed_cols = df.columns[df.columns.str.contains('^Unnamed')]
    if not unnamed_cols.empty:
        # Create new column names for unnamed columns
        new_names = {col: f'Feature_{i+1}' for i, col in enumerate(unnamed_cols)}
        df = df.rename(columns=new_names)
        st.info(f"Renamed {len(unnamed_cols)} unnamed columns to Feature_N format")
    
    return df

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return list of numeric columns from DataFrame."""
    return df.select_dtypes(include=['float64', 'int64']).columns.tolist()

def needs_cleaning(df: pd.DataFrame, feature_columns: List[str], target_column: str) -> Tuple[bool, str]:
    """
    Check if the dataset needs cleaning by analyzing missing values and data quality.
    
    Returns:
    - bool: True if cleaning is needed, False otherwise
    - str: Description of why cleaning is needed (if True)
    """
    try:
        # Check if all columns exist
        missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
        if missing_cols:
            return True, f"Missing columns: {missing_cols}"
            
        # Get feature and target data
        X = df[feature_columns]
        y = df[target_column]
        
        # Check for missing values
        feature_nulls = X.isnull().sum().sum()
        target_nulls = y.isnull().sum()
        
        if feature_nulls > 0 or target_nulls > 0:
            return True, f"Found {feature_nulls} missing values in features and {target_nulls} in target"
            
        # Check for completely empty columns
        empty_cols = X.columns[X.isnull().all()].tolist()
        if empty_cols:
            return True, f"Found completely empty columns: {empty_cols}"
            
        # Check for infinite values
        inf_count = np.isinf(X.select_dtypes(include=np.number)).sum().sum()
        if inf_count > 0:
            return True, f"Found {inf_count} infinite values in features"
            
        return False, "Data is clean"
        
    except Exception as e:
        return True, f"Error checking data: {str(e)}"

def clean_dataset(df: pd.DataFrame, 
                 feature_columns: List[str], 
                 target_column: str,
                 feature_imputation_strategy: str = 'mean',
                 target_imputation_strategy: str = 'drop',
                 knn_neighbors: int = 5,
                 force_cleaning: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean the dataset by handling missing values. Only cleans if needed or if force_cleaning is True.
    """
    try:
        # Check if cleaning is needed
        needs_clean, reason = needs_cleaning(df, feature_columns, target_column)
        
        if not needs_clean and not force_cleaning:
            # If data is already clean and not forcing cleaning, return as is
            return df[feature_columns].copy(), df[target_column].copy()
            
        if needs_clean:
            st.info(f"Data cleaning needed: {reason}")
        elif force_cleaning:
            st.info("Forced data cleaning mode")

        # Validate input columns
        if not all(col in df.columns for col in feature_columns):
            missing_cols = [col for col in feature_columns if col not in df.columns]
            raise ValueError(f"Feature columns not found in dataset: {missing_cols}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Create copies to avoid modifying original data
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Check for completely empty columns
        empty_cols = X.columns[X.isnull().all()].tolist()
        if empty_cols:
            st.warning(f"Found completely empty columns: {empty_cols}")
            # Remove empty columns from feature set
            feature_columns = [col for col in feature_columns if col not in empty_cols]
            X = X.drop(columns=empty_cols)
            if not feature_columns:
                raise ValueError("No valid features remaining after removing empty columns")
        
        # Count missing values
        feature_nulls = X.isnull().sum().sum()
        target_nulls = y.isnull().sum()
        
        if feature_nulls > 0:
            st.info(f"Found {feature_nulls} missing values in features.")
            
            if feature_imputation_strategy == 'drop':
                # Drop rows with any missing values in features
                mask = X.notna().all(axis=1)
                X = X[mask]
                y = y[mask]
                st.warning(f"Dropped {feature_nulls} rows with missing feature values.")
            else:
                # Impute missing values in features
                if feature_imputation_strategy == 'knn':
                    imputer = KNNImputer(n_neighbors=min(knn_neighbors, len(X)-1))
                    X_imputed = imputer.fit_transform(X)
                    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
                else:
                    for col in X.columns:
                        # Handle each column separately to avoid the "no observed values" warning
                        col_data = X[col]
                        if col_data.isnull().all():
                            # If column is all null, fill with 0
                            X[col] = 0
                        else:
                            imputer = SimpleImputer(strategy=feature_imputation_strategy)
                            X[col] = imputer.fit_transform(col_data.values.reshape(-1, 1)).ravel()
                
                st.success(f"Imputed {feature_nulls} missing feature values using {feature_imputation_strategy}.")
        
        if target_nulls > 0:
            st.info(f"Found {target_nulls} missing values in target variable.")
            
            if target_imputation_strategy == 'drop':
                # Drop rows with missing target values
                mask = y.notna()
                X = X[mask]
                y = y[mask]
                st.warning(f"Dropped {target_nulls} rows with missing target values.")
            else:
                # Impute missing values in target
                if target_imputation_strategy == 'knn':
                    # Reshape y for KNN imputation
                    y_2d = y.values.reshape(-1, 1)
                    imputer = KNNImputer(n_neighbors=min(knn_neighbors, len(y)-1))
                    y = pd.Series(
                        imputer.fit_transform(y_2d).ravel(),
                        index=y.index,
                        name=y.name
                    )
                else:
                    if y.isnull().all():
                        # If target is all null, fill with 0
                        y = pd.Series(0, index=y.index, name=y.name)
                    else:
                        imputer = SimpleImputer(strategy=target_imputation_strategy)
                        y = pd.Series(
                            imputer.fit_transform(y.values.reshape(-1, 1)).ravel(),
                            index=y.index,
                            name=y.name
                        )
                st.success(f"Imputed {target_nulls} missing target values using {target_imputation_strategy}.")
        
        # Ensure X and y have the same length
        if len(X) != len(y):
            raise ValueError(f"Feature and target lengths don't match: X={len(X)}, y={len(y)}")
        
        # Final check for any remaining missing values
        if X.isnull().any().any() or y.isnull().any():
            raise ValueError("Data still contains missing values after cleaning.")
        
        return X, y
        
    except Exception as e:
        st.error(f"Error during data cleaning: {str(e)}")
        return None, None

def compute_sdr(y: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> float:
    """
    Compute Standard Deviation Reduction for a split.
    SDR = std_dev(parent) - weighted_avg(std_dev(children))
    """
    parent_std = np.std(y)
    
    left_std = np.std(y[left_indices]) if len(left_indices) > 0 else 0
    right_std = np.std(y[right_indices]) if len(right_indices) > 0 else 0
    
    n = len(y)
    n_left = len(left_indices)
    n_right = len(right_indices)
    
    weighted_children_std = (n_left/n * left_std) + (n_right/n * right_std)
    
    return parent_std - weighted_children_std

@st.cache_data
def train_model(X, y, min_samples_leaf=1, max_depth=5, test_size=0.2):
    """Train the decision tree model with proper error handling."""
    try:
        if X is None or y is None:
            raise ValueError("Input data is None. Data cleaning may have failed.")
            
        start_time = time.time()
        
        # Convert to numpy arrays
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y = y.to_numpy() if isinstance(y, pd.Series) else y
        
        # Split the data
        if len(X) < 2:
            raise ValueError("Not enough samples for training after cleaning.")
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Create and train the model
        model = DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        training_time = time.time() - start_time
        
        return model, {
            'train_score': train_score,
            'test_score': test_score
        }, training_time
        
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        return None, None, None

def create_tree_visualization(model: DecisionTreeRegressor, 
                           feature_names: List[str],
                           orientation: str = "horizontal") -> str:
    """Create a graphviz visualization of the decision tree.
    Returns the DOT source code as a string for Streamlit compatibility.
    
    Parameters:
    -----------
    model : DecisionTreeRegressor
        The trained decision tree model
    feature_names : List[str]
        List of feature names
    orientation : str
        Tree orientation, either "horizontal" or "vertical"
    """
    dot_data = export_graphviz(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        rotate=True if orientation == "horizontal" else False,
        proportion=True
    )
    
    # Add responsive styling to the DOT source with dark theme
    dot_data = dot_data.replace(
        'digraph Tree {',
        '''digraph Tree {
        // Dark theme and responsive styling
        bgcolor="transparent";
        graph [fontname="Arial" ranksep=0.3 nodesep=0.3];
        node [fontname="Arial" fontsize="12" style="filled, rounded" color="#30363d" fontcolor="black"];
        edge [fontname="Arial" fontsize="10" color="#6e7681"];'''
    )
    
    # Replace default colors with dark theme colors and ensure text contrast
    dot_data = dot_data.replace('fillcolor="#e5813900"', 'fillcolor="#1a1b26" fontcolor="black"')  # Darkest nodes
    dot_data = dot_data.replace('fillcolor="#e5813999"', 'fillcolor="#1a1b26" fontcolor="black"')  # Medium nodes
    dot_data = dot_data.replace('fillcolor="#e58139ff"', 'fillcolor="#1a1b26" fontcolor="black"')  # Lightest nodes
    
    # Make sure all text is visible by adding contrast
    dot_data = dot_data.replace('label=<', 'fontcolor="black" label=<')
    
    return dot_data

def get_node_data(model: DecisionTreeRegressor, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Extract node information from the trained model."""
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    
    node_data = []
    for node_id in range(n_nodes):
        if children_left[node_id] != children_right[node_id]:  # Internal node
            split_feature = X.columns[feature[node_id]]
            split_threshold = f"{threshold[node_id]:.4f}"
        else:
            split_feature = "leaf"
            split_threshold = "-"
            
        node_data.append({
            'node_id': node_id,
            'samples': model.tree_.n_node_samples[node_id],
            'mean_target': model.tree_.value[node_id][0][0],
            'split_feature': split_feature,
            'split_threshold': split_threshold
        })
    
    return pd.DataFrame(node_data)

def predict_single_instance(
    model: DecisionTreeRegressor,
    input_data: Dict[str, float],
    feature_names: List[str]
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Make a prediction for a single instance and return the decision path.
    """
    # Convert input dictionary to array
    X = np.array([input_data[f] for f in feature_names]).reshape(1, -1)
    
    # Get prediction
    prediction = model.predict(X)[0]
    
    # Get decision path
    decision_path = model.decision_path(X)
    path_indices = decision_path.indices
    
    # Extract path information
    path_info = []
    for node_idx in path_indices:
        if model.tree_.feature[node_idx] >= 0:  # Not a leaf
            feature_name = feature_names[model.tree_.feature[node_idx]]
            threshold = model.tree_.threshold[node_idx]
            value = input_data[feature_name]
            path_info.append({
                'node': node_idx,
                'feature': feature_name,
                'threshold': threshold,
                'value': value,
                'decision': 'left' if value <= threshold else 'right'
            })
        else:  # Leaf node
            path_info.append({
                'node': node_idx,
                'feature': 'leaf',
                'prediction': model.tree_.value[node_idx][0][0]
            })
    
    return prediction, path_info

def export_model_config(
    model: DecisionTreeRegressor,
    feature_names: List[str],
    target_name: str,
    hyperparameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Export model configuration and parameters."""
    return {
        'features': feature_names,
        'target': target_name,
        'hyperparameters': hyperparameters,
        'tree_structure': {
            'n_nodes': model.tree_.node_count,
            'max_depth': model.get_depth(),
            'feature_importance': dict(zip(feature_names, model.feature_importances_.tolist()))
        }
    } 