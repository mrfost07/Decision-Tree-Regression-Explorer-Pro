import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    load_data, get_numeric_columns, train_model, create_tree_visualization,
    get_node_data, predict_single_instance, export_model_config, compute_sdr,
    clean_dataset, needs_cleaning
)
import pickle
import json
import time
from io import BytesIO

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Decision Tree Regression Explorer Pro",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state for tab selection
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

# Main header with simplified styling
st.markdown("""
    <div class='simple-header'>
        <h1>🌳 Decision Tree Regression Explorer Pro</h1>
    </div>
""", unsafe_allow_html=True)

# Add description with typing effect using st.empty()
description_placeholder = st.empty()
description_placeholder.markdown("""
    <p class="typing-text">A professional-grade tool for interactive decision tree analysis and prediction</p>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'target_name' not in st.session_state:
    st.session_state.target_name = None
if 'training_time' not in st.session_state:
    st.session_state.training_time = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Main tabs with enhanced styling
tabs = st.tabs([
    "📊 Data Explorer",
    "🔍 Model Training",
    "🎯 Predictions",
    "ℹ️ Documentation"
])

# Sidebar with enhanced organization
with st.sidebar:
    st.markdown("""
        <div class='stCard'>
            <h2>Model Configuration</h2>
            <p class="subtitle">Configure your model parameters and data settings</p>
        </div>
    """, unsafe_allow_html=True)
    
    # File upload with enhanced UI
    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)",
        type=['csv'],
        help="Upload your dataset in CSV format"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing data..."):
            df = load_data(uploaded_file)
            numeric_cols = get_numeric_columns(df)
            
            st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Enhanced feature selection UI
            st.markdown("""
                <div class='stCard'>
                    <h3>Feature Selection</h3>
                    <p class="subtitle">Choose the variables for your model</p>
                </div>
            """, unsafe_allow_html=True)
            
            selected_features = st.multiselect(
                "Select Features",
                options=numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols,
                help="Choose the features to use for prediction"
            )
            
            available_targets = [col for col in numeric_cols if col not in selected_features]
            target_variable = st.selectbox(
                "Target Variable",
                options=available_targets,
                help="Choose the variable to predict"
            )
            
            # Check if data needs cleaning
            needs_clean, reason = needs_cleaning(df, selected_features, target_variable)
            
            # Enhanced data cleaning options - only show if needed
            if needs_clean:
                st.markdown(f"""
                    <div class='stCard warning-card'>
                        <h3>⚠️ Data Preprocessing Needed</h3>
                        <p class="subtitle">{reason}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                    <div class='stCard'>
                        <h3>Data Preprocessing</h3>
                        <p class="subtitle">Configure how to handle data quality issues</p>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    feature_imputation = st.selectbox(
                        "Feature Imputation",
                        options=['mean', 'median', 'most_frequent', 'knn', 'drop'],
                        help="Strategy for handling missing values in features"
                    )
                
                with col2:
                    target_imputation = st.selectbox(
                        "Target Imputation",
                        options=['drop', 'mean', 'median', 'most_frequent', 'knn'],
                        help="Strategy for handling missing values in target"
                    )
                
                if 'knn' in [feature_imputation, target_imputation]:
                    knn_neighbors = st.slider(
                        "KNN Neighbors",
                        min_value=1,
                        max_value=20,
                        value=5,
                        help="Number of neighbors for KNN imputation"
                    )
                else:
                    knn_neighbors = 5
                    
                force_cleaning = False
            else:
                st.markdown("""
                    <div class='stCard success-card'>
                        <h3>✅ Data Quality Check Passed</h3>
                        <p class="subtitle">No preprocessing needed - your data is clean!</p>
                    </div>
                """, unsafe_allow_html=True)
                
                force_cleaning = st.checkbox(
                    "Force Data Preprocessing",
                    value=False,
                    help="Enable this if you want to apply preprocessing even though data appears clean"
                )
                
                if force_cleaning:
                    col1, col2 = st.columns(2)
                    with col1:
                        feature_imputation = st.selectbox(
                            "Feature Imputation",
                            options=['mean', 'median', 'most_frequent', 'knn', 'drop'],
                            help="Strategy for handling missing values in features"
                        )
                    
                    with col2:
                        target_imputation = st.selectbox(
                            "Target Imputation",
                            options=['drop', 'mean', 'median', 'most_frequent', 'knn'],
                            help="Strategy for handling missing values in target"
                        )
                    
                    if 'knn' in [feature_imputation, target_imputation]:
                        knn_neighbors = st.slider(
                            "KNN Neighbors",
                            min_value=1,
                            max_value=20,
                            value=5,
                            help="Number of neighbors for KNN imputation"
                        )
                    else:
                        knn_neighbors = 5
                else:
                    feature_imputation = 'mean'
                    target_imputation = 'drop'
                    knn_neighbors = 5
            
            # Enhanced model parameters
            st.markdown("""
                <div class='stCard'>
                    <h3>Model Parameters</h3>
                    <p class="subtitle">Fine-tune your decision tree</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                min_samples_leaf = st.slider(
                    "Min Samples per Leaf",
                    min_value=1,
                    max_value=20,
                    value=1,
                    help="Minimum samples required at leaf nodes"
                )
            
            with col2:
                max_depth = st.slider(
                    "Max Tree Depth",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Maximum depth of the tree"
                )
            
            test_size = st.slider(
                "Test Set Size",
                min_value=10,
                max_value=50,
                value=20,
                help="Percentage of data to use for testing"
            ) / 100
            
            # Function to switch to Model Training tab
            def switch_to_model_training():
                st.session_state.current_tab = 1

            # Enhanced training button
            train_button = st.button(
                "🚀 Train Model",
                disabled=not (selected_features and target_variable),
                help="Start model training with current configuration",
                on_click=switch_to_model_training
            )
        
        if train_button:
            with st.spinner("Cleaning data and training model..."):
                try:
                    # Clean the data first
                    X, y = clean_dataset(
                        df,
                        selected_features,
                        target_variable,
                        feature_imputation_strategy=feature_imputation,
                        target_imputation_strategy=target_imputation,
                        knn_neighbors=knn_neighbors,
                        force_cleaning=force_cleaning
                    )
                    
                    if X is None or y is None:
                        st.error("Data cleaning failed. Please check your data and try different cleaning options.")
                        st.session_state.model = None
                        st.session_state.metrics = None
                        st.session_state.training_time = None
                    else:
                        # Train the model with cleaned data
                        model, metrics, training_time = train_model(
                            X, y,
                            min_samples_leaf=min_samples_leaf,
                            max_depth=max_depth,
                            test_size=test_size
                        )
                        
                        if model is not None and metrics is not None and training_time is not None:
                            st.session_state.model = model
                            st.session_state.feature_names = selected_features
                            st.session_state.target_name = target_variable
                            st.session_state.training_time = training_time
                            st.session_state.metrics = metrics
                            
                            # Performance metrics in cards
                            st.markdown("""
                                <div class='metric-card'>
                                    <h4>Training Results</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.metric("⏱️ Training Time", f"{training_time:.2f}s")
                            st.metric("📈 Training R²", f"{metrics['train_score']:.4f}")
                            st.metric("🎯 Testing R²", f"{metrics['test_score']:.4f}")
                        else:
                            st.error("Model training failed. Please check your data and try again.")
                            st.session_state.model = None
                            st.session_state.metrics = None
                            st.session_state.training_time = None
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.session_state.model = None
                    st.session_state.metrics = None
                    st.session_state.training_time = None

# Data Preview tab with enhanced visualization
with tabs[0]:
    if uploaded_file is not None:
        st.markdown("""
            <div class='stCard'>
                <h2>Dataset Overview</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Interactive data preview
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
        
        # Dataset statistics in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Numeric Columns", len(numeric_cols))
        
        # Detailed statistics
        with st.expander("📊 Detailed Statistics", expanded=True):
            st.dataframe(
                df[numeric_cols].describe(),
                use_container_width=True
            )

# Train & Visualize tab with interactive components
with tabs[1]:
    if st.session_state.model is not None:
        st.markdown("""
            <div class='stCard'>
                <h2>Model Analysis</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
                <div class='stCard visualization-container'>
                    <h3>Decision Tree Visualization</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Visualization controls
            viz_col1, viz_col2 = st.columns([2, 1])
            with viz_col2:
                zoom_level = st.slider(
                    "Zoom Level",
                    min_value=50,
                    max_value=150,
                    value=100,
                    step=10,
                    help="Adjust the size of the visualization"
                )
                
                orientation = st.radio(
                    "Tree Orientation",
                    options=["horizontal", "vertical"],
                    index=0,
                    help="Choose the tree layout direction"
                )
            
            with viz_col1:
                # Get the DOT source for the tree visualization
                dot_source = create_tree_visualization(
                    st.session_state.model,
                    st.session_state.feature_names,
                    orientation=orientation
                )
                
                # Render the graph using Streamlit's graphviz_chart
                if dot_source:
                    try:
                        st.markdown(f"""
                            <div class="tree-visualization" style="zoom: {zoom_level}%;">
                        """, unsafe_allow_html=True)
                        st.graphviz_chart(dot_source, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error rendering tree visualization: {str(e)}")
                        st.code(dot_source, language="dot")
                else:
                    st.warning("Could not generate tree visualization.")
        
        with col2:
            st.markdown("""
                <div class='stCard'>
                    <h3>Node Information</h3>
                </div>
            """, unsafe_allow_html=True)
            
            node_df = get_node_data(
                st.session_state.model,
                df[st.session_state.feature_names],
                df[st.session_state.target_name]
            )
            st.dataframe(node_df, use_container_width=True)
        
        # Export options with enhanced UI
        st.markdown("""
            <div class='stCard'>
                <h3>Export Options</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_bytes = pickle.dumps(st.session_state.model)
            st.download_button(
                "📥 Download Model (pickle)",
                model_bytes,
                "decision_tree_model.pkl",
                "application/octet-stream",
                help="Download the trained model for later use"
            )
        
        with col2:
            config = export_model_config(
                st.session_state.model,
                st.session_state.feature_names,
                st.session_state.target_name,
                {
                    'min_samples_leaf': min_samples_leaf,
                    'max_depth': max_depth,
                    'test_size': test_size
                }
            )
            config_bytes = json.dumps(config, indent=2).encode()
            st.download_button(
                "💾 Save Configuration (JSON)",
                config_bytes,
                "model_config.json",
                "application/json",
                help="Save model configuration and parameters"
            )

# Predict tab with interactive features
with tabs[2]:
    if st.session_state.model is not None:
        st.markdown("""
            <div class='stCard'>
                <h2>Make Predictions</h2>
                <p>Enter feature values to get predictions from the model</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Create input fields in columns for better organization
        cols = st.columns(3)
        input_data = {}
        
        for idx, feature in enumerate(st.session_state.feature_names):
            with cols[idx % 3]:
                input_data[feature] = st.number_input(
                    f"Enter {feature}",
                    value=float(df[feature].mean()),
                    format="%.4f",
                    help=f"Average value: {df[feature].mean():.2f}"
                )
        
        # Prediction button with loading state
        if st.button("🎯 Predict", help="Get prediction based on input values"):
            with st.spinner("Computing prediction..."):
                prediction, path_info = predict_single_instance(
                    st.session_state.model,
                    input_data,
                    st.session_state.feature_names
                )
            
            # Display prediction result in a card
            st.markdown("""
                <div class='metric-card'>
                    <h3>Prediction Result</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.metric(
                f"Predicted {st.session_state.target_name}",
                f"{prediction:.4f}"
            )
            
            # Show decision path with enhanced visualization
            st.markdown("""
                <div class='stCard'>
                    <h3>Decision Path Analysis</h3>
                </div>
            """, unsafe_allow_html=True)
            
            for step in path_info[:-1]:
                st.markdown(f"""
                    <div class='metric-card'>
                        <p>Node {step['node']}: Is {step['feature']} ≤ {step['threshold']:.4f}?</p>
                        <p>Value: {step['value']:.4f} → {step['decision']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            leaf = path_info[-1]
            st.markdown(f"""
                <div class='metric-card'>
                    <p>🎯 Final Prediction at Leaf Node {leaf['node']}: {leaf['prediction']:.4f}</p>
                </div>
            """, unsafe_allow_html=True)

# About tab with comprehensive documentation
with tabs[3]:
    st.markdown("""
        <div class='stCard'>
            <h2>About Decision Tree Regression Explorer Pro</h2>
            <p>A professional-grade tool for interactive decision tree analysis and prediction</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📚 What is Decision Tree Regression?", expanded=True):
        st.write("""
        Decision Tree Regression is a powerful machine learning algorithm that predicts continuous target variables by learning decision rules from features. It creates a tree-like structure of decisions, making it both powerful and interpretable.
        
        Key advantages:
        - 📊 Easy to understand and interpret
        - 🔄 Can handle both numerical and categorical data
        - 🚀 Requires minimal data preprocessing
        - 📈 Can capture non-linear relationships
        """)
    
    with st.expander("📐 Standard Deviation Reduction (SDR)", expanded=False):
        st.write("""
        SDR measures the reduction in standard deviation achieved by a split. The formula is:
        
        ```
        SDR = std_dev(parent) - weighted_avg(std_dev(children))
        ```
        
        where:
        - std_dev(parent) is the standard deviation of the target variable in the parent node
        - weighted_avg(std_dev(children)) is the weighted average of the standard deviations in the child nodes
        """)
    
    with st.expander("🛑 Stopping Criteria", expanded=False):
        st.write("""
        The tree stops growing when one of these criteria is met:
        1. Maximum depth is reached
        2. Number of samples in a node is less than min_samples_leaf
        3. All samples in a node have the same target value
        4. No split would reduce the standard deviation
        """)
    
    with st.expander("🔗 Useful Resources", expanded=False):
        st.markdown("""
        - [Scikit-learn Decision Tree Regressor Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
        - [Understanding Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
        - [Streamlit Documentation](https://docs.streamlit.io)
        
        For support or feature requests, please visit our [GitHub repository](https://github.com/yourusername/decision-tree-explorer).
        """)

# Documentation tab content ends here

# Add enhanced footer
st.markdown("""
    <div class='footer'>
        <p>
            © 2025 Decision Tree Regression Explorer Pro • Made with ❤️ by 
            <a href='https://github.com/mrfost07' target='_blank'>Renier Fostanes</a>
        </p>
    </div>
""", unsafe_allow_html=True) 