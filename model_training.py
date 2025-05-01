import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from io import StringIO
import pydotplus
import os

# Create directory for model outputs
os.makedirs('model_outputs', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

def load_and_prepare_data(filepath):
    """Load and prepare data for model training"""
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}.  Please make sure the file exists and the path is correct.")
        return None, None, None, None, None, None, None  # Return None values to indicate failure
    
    # Print dataset information
    print(f"Dataset shape: {df.shape}")
    print("\nAvailable columns:", df.columns.tolist())
    
    # Feature selection
    # Note: Adjust these features based on your dataset
    features = ['NDVI_N', 'Slope_N', 'Water_N', 'Land_N']  #These are just example names, change it as per your file
    target = 'Conflict_R' #This is just an example name, change it as per your file
    
    # Verify columns exist
    print(f"\nTarget column exists: {target in df.columns}")
    if target not in df.columns:
        print(f"Error: Target column '{target}' not found in the dataset.  Please check the target column name.")
        return None, None, None, None, None, None, None
    
    for feature in features:
        print(f"Feature column {feature} exists: {feature in df.columns}")
        if feature not in df.columns:
            print(f"Error: Feature column '{feature}' not found in the dataset.  Please check the feature column names.")
            return None, None, None, None, None, None, None
    
    # Handle missing values
    print("\nMissing values in selected columns:")
    print(df[features + [target]].isnull().sum())
    
    # Impute missing values if needed
    if df[features + [target]].isnull().sum().sum() > 0:
        print("Imputing missing values...")
        for col in features + [target]:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Check for class imbalance
    print("\nClass distribution:")
    print(df[target].value_counts(normalize=True))
    
    # Prepare data
    X = df[features]
    y = df[target]
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    joblib.dump(scaler, 'model_outputs/feature_scaler.pkl')
    
    print("Data preparation complete.")
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, features

def train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, features):
    """Train and evaluate multiple models"""
    if X_train is None:
        print("Error: No data to train models.  Exiting.")
        return None, None
    print("\n=== Training and Evaluating Models ===")
    
    # Models to train
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    # Parameter grids for each model
    param_grids = {
        'Decision Tree': {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 8]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01]
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l1', 'l2']
        }
    }
    
    # Dictionary to store best models
    best_models = {}
    best_scores = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Decide if this model needs scaled features
        use_scaled = name in ['SVM', 'Neural Network', 'Logistic Regression']
        X_train_current = X_train_scaled if use_scaled else X_train
        X_test_current = X_test_scaled if use_scaled else X_test
        
        # Perform grid search
        grid_search = GridSearchCV(
            model,
            param_grids[name],
            cv=5,
            scoring='f1_weighted',
            verbose=1
        )
        
        grid_search.fit(X_train_current, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_current)
        
        # Calculate metrics
        print(f"\n{name} - Best Parameters: {grid_search.best_params_}")
        print("\nClassification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))
        
        # Store F1 score
        best_scores[name] = report['weighted avg']['f1-score']
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[f'Class {i}' for i in sorted(y_test.unique())],
                        yticklabels=[f'Class {i}' for i in sorted(y_test.unique())])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(f'visualizations/confusion_matrix_{name.replace(" ", "_")}.png')
        
        # ROC Curve if applicable
        if hasattr(best_model, "predict_proba") and len(y_test.unique()) == 2: # Only for binary classification
            try:
                y_scores = best_model.predict_proba(X_test_current)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {name}')
                plt.legend(loc='lower right')
                plt.savefig(f'visualizations/roc_curve_{name.replace(" ", "_")}.png')
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test, y_scores)
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - {name}')
                plt.savefig(f'visualizations/pr_curve_{name.replace(" ", "_")}.png')
            except Exception as e:
                print(f"Error generating ROC/PR curve: {e}")
        elif hasattr(best_model, "predict_proba") and len(y_test.unique()) > 2:
             print(f"Skipping ROC and Precision-Recall curves for {name} as it is a multiclass problem.")
        
        # Save model
        joblib.dump(best_model, f'model_outputs/{name.replace(" ", "_")}_model.pkl')
    
    # Compare models
    print("\n=== Model Comparison ===")
    for name, score in sorted(best_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: F1 Score = {score:.4f}")
    
    # Identify best model
    best_model_name = max(best_scores, key=best_scores.get)
    print(f"\nBest Model: {best_model_name} (F1 Score: {best_scores[best_model_name]:.4f})")
    
    # Save the best model as the main model
    joblib.dump(best_models[best_model_name], 'model_outputs/conflict_risk_model.pkl')
    print(f"Best model saved as 'model_outputs/conflict_risk_model.pkl'")
    
    # If the best model is a tree-based model, visualize it
    if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
        visualize_feature_importance(best_models[best_model_name], features, best_model_name)
        
        # If it's a Decision Tree, visualize the tree
        if best_model_name == 'Decision Tree':
            try:
                dot_data = StringIO()
                export_graphviz(best_models[best_model_name], out_file=dot_data, 
                                 feature_names=features,
                                 filled=True, rounded=True,
                                 special_characters=True)
                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                graph.write_png('visualizations/decision_tree.png')
                print("Decision tree visualization saved as 'visualizations/decision_tree.png'")
            except Exception as e:
                print(f"Couldn't visualize tree: {str(e)}")
    
    return best_models[best_model_name], best_model_name

def visualize_feature_importance(model, features, model_name):
    """Visualize feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        indices = np.argsort(feature_importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance - {model_name}")
        plt.bar(range(len(features)), feature_importance[indices], align='center')
        plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'visualizations/feature_importance_{model_name.replace(" ", "_")}.png')
        print(f"Feature importance saved as 'visualizations/feature_importance_{model_name.replace(" ", "_")}.png'")

def main():
    """Main function to train and evaluate models"""
    # Load and prepare data
    # Replace "enhanced_wildlife_data.csv" with the actual path to your CSV file
    filepath = "wsm_output.csv"  # <--- CHANGE THIS LINE TO YOUR FILE NAME
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, features = load_and_prepare_data(filepath)
    
    # Check if data loading was successful
    if X_train is not None:
        # Train and evaluate models
        best_model, best_model_name = train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, features)
        
        print("\nModel training and evaluation complete.")
        return best_model, best_model_name
    else:
        print("Error: Data loading failed.  Exiting.")
        return None, None

if __name__ == "__main__":
    main()
