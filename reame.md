# Project Structure
The improved Human-Animal Conflict project is organized into the following components:

1. `model_training.py` - Enhanced model training with multiple algorithms
2. `iot_simulator.py` - Improved IoT data simulation
3. `data_preprocessing.py` - Robust data preprocessing pipeline
4. `visualization.py` - Enhanced visualization with grid lines
5. `dashboard.py` - Improved Streamlit dashboard
6. `utils.py` - Utility functions and constants

# 1. model_training.py
```python
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
    df = pd.read_csv(filepath)
    
    # Print dataset information
    print(f"Dataset shape: {df.shape}")
    print("\nAvailable columns:", df.columns.tolist())
    
    # Feature selection
    # Note: Adjust these features based on your dataset
    features = ['NDVI_N', 'Slope_N', 'Water_N', 'Land_N']
    target = 'Conflict_R'
    
    # Verify columns exist
    print(f"\nTarget column exists: {target in df.columns}")
    for feature in features:
        print(f"Feature column {feature} exists: {feature in df.columns}")
    
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
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(f'visualizations/confusion_matrix_{name.replace(" ", "_")}.png')
        
        # ROC Curve if applicable
        if hasattr(best_model, "predict_proba"):
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
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, features = load_and_prepare_data("enhanced_wildlife_data.csv")
    
    # Train and evaluate models
    best_model, best_model_name = train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, features)
    
    print("\nModel training and evaluation complete.")
    return best_model, best_model_name

if __name__ == "__main__":
    main()
```

# 2. iot_simulator.py
```python
import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import random
from datetime import datetime, timedelta

class IoTSimulator:
    """Class to simulate IoT data for wildlife conflict prediction"""
    
    def __init__(self, model_path='model_outputs/conflict_risk_model.pkl', scaler_path='model_outputs/feature_scaler.pkl', simulation_type='random'):
        """Initialize the IoT simulator"""
        # Load model and scaler
        print("Loading model and scaler...")
        self.model = joblib.load(model_path)
        
        try:
            self.scaler = joblib.load(scaler_path)
            self.use_scaler = True
        except:
            self.use_scaler = False
            print("No scaler found. Using raw values.")
        
        # Get feature names from the model
        if hasattr(self.model, 'feature_names_in_'):
            self.features = self.model.feature_names_in_
        else:
            # Fallback to default features if not available in model
            self.features = ['NDVI_N', 'Slope_N', 'Water_N', 'Land_N']
        
        print(f"Using features: {self.features}")
        
        # History storage
        self.data_history = []
        self.prediction_history = []
        self.timestamp_history = []
        
        # Simulation parameters
        self.simulation_type = simulation_type  # 'random', 'seasonal', 'spatial'
        self.current_time = datetime.now()
        
        # For seasonal simulation
        self.seasonal_factor = 0
        self.seasonal_direction = 1
        
        # For spatial simulation
        self.spatial_x = 0.5
        self.spatial_y = 0.5
        self.spatial_direction_x = 0.05
        self.spatial_direction_y = 0.03
        
        print(f"IoT simulator initialized with {simulation_type} simulation type")
    
    def generate_data(self):
        """Generate simulated IoT sensor data"""
        # Base data dictionary
        data = {}
        
        if self.simulation_type == 'random':
            # Completely random data
            for feature in self.features:
                data[feature] = np.round(np.random.uniform(0, 1), 2)
                
        elif self.simulation_type == 'seasonal':
            # Seasonal pattern (simulating day/night or seasonal changes)
            self.seasonal_factor += 0.1 * self.seasonal_direction
            if self.seasonal_factor > 1 or self.seasonal_factor < 0:
                self.seasonal_direction *= -1
                self.seasonal_factor += 0.2 * self.seasonal_direction
            
            # Different seasonal patterns for different features
            for feature in self.features:
                if feature == 'NDVI_N':
                    # NDVI changes with seasons
                    data[feature] = np.round(0.3 + 0.6 * self.seasonal_factor, 2)
                elif feature == 'Water_N':
                    # Water availability changes with rainfall
                    data[feature] = np.round(0.2 + 0.7 * (1 - self.seasonal_factor), 2)
                else:
                    # Other features have smaller variations
                    data[feature] = np.round(0.3 + 0.4 * np.random.uniform(0, 1), 2)
                    
        elif self.simulation_type == 'spatial':
            # Spatial movement pattern (simulating animal movement)
            self.spatial_x += self.spatial_direction_x
            self.spatial_y += self.spatial_direction_y
            
            # Bounce at boundaries
            if self.spatial_x > 1 or self.spatial_x < 0:
                self.spatial_direction_x *= -1
                self.spatial_x += self.spatial_direction_x
            if self.spatial_y > 1 or self.spatial_y < 0:
                self.spatial_direction_y *= -1
                self.spatial_y += self.spatial_direction_y
            
            # Translate spatial position to sensor readings
            for feature in self.features:
                if feature == 'NDVI_N':
                    # Higher NDVI in some areas
                    data[feature] = np.round(0.2 + 0.7 * np.sin(self.spatial_x * np.pi), 2)
                elif feature == 'Water_N':
                    # Water proximity changes with position
                    data[feature] = np.round(0.1 + 0.8 * np.cos(self.spatial_y * np.pi), 2)
                elif feature == 'Slope_N':
                    # Slope varies across landscape
                    data[feature] = np.round(0.3 + 0.6 * np.sin(self.spatial_x * np.pi + self.spatial_y * np.pi), 2)
                else:
                    # Other features
                    data[feature] = np.round(0.2 + 0.6 * np.random.uniform(0, 1), 2)
        
        # Add small noise to all values
        for feature in data:
            noise = np.random.uniform(-0.05, 0.05)
            data[feature] = max(0, min(1, data[feature] + noise))
            data[feature] = np.round(data[feature], 2)
        
        # Add timestamp
        self.current_time += timedelta(minutes=5)
        
        return data
    
    def predict_conflict(self, data):
        """Predict conflict risk from sensor data"""
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Filter to include only the features the model expects
        df = df[self.features]
        
        # Scale data if needed
        if self.use_scaler:
            df_scaled = self.scaler.transform(df)
            prediction = self.model.predict(df_scaled)[0]
            # Get probability if available
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(df_scaled)[0]
                probability = probabilities[1]  # Probability of class 1
            else:
                probability = None
        else:
            prediction = self.model.predict(df)[0]
            # Get probability if available
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(df)[0]
                probability = probabilities[1]  # Probability of class 1
            else:
                probability = None
        
        return prediction, probability
    
    def run_simulation(self, iterations=100, delay=1):
        """Run the simulation for a number of iterations"""
        print(f"Starting {self.simulation_type} simulation for {iterations} iterations...")
        
        for i in range(iterations):
            # Generate data
            data = self.generate_data()
            
            # Predict conflict
            prediction, probability = self.predict_conflict(data)
            
            # Store data for visualization
            self.data_history.append(data)
            self.prediction_history.append(prediction)
            self.timestamp_history.append(self.current_time)
            
            # Print results
            risk_level = "High" if prediction == 1 else "Low"
            prob_str = f" (Probability: {probability:.2f})" if probability is not None else ""
            print(f"Iteration {i+1}/{iterations}: {risk_level} Risk{prob_str}")
            print(f"Sensor Data: {data}")
            print("-" * 50)
            
            # Delay
            time.sleep(delay)
        
        print("Simulation complete.")
    
    def visualize_history(self):
        """Visualize the simulation history"""
        if not self.data_history:
            print("No simulation data available to visualize.")
            return
        
        # Convert histories to DataFrames
        history_df = pd.DataFrame(self.data_history)
        history_df['timestamp'] = self.timestamp_history
        history_df['prediction'] = self.prediction_history
        
        # Set up plot
        plt.figure(figsize=(15, 10))
        
        # Plot sensor data over time
        plt.subplot(2, 1, 1)
        for feature in self.features:
            plt.plot(history_df.index, history_df[feature], label=feature)
        plt.xlabel('Time Steps')
        plt.ylabel('Sensor Values')
        plt.title('Sensor Data Over Time')
        plt.legend()
        plt.grid(True)
        
        # Plot predictions over time
        plt.subplot(2, 1, 2)
        plt.plot(history_df.index, history_df['prediction'], 'ro-', label='Conflict Risk (1=High, 0=Low)')
        plt.xlabel('Time Steps')
        plt.ylabel('Prediction')
        plt.title('Conflict Risk Predictions Over Time')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizations/simulation_history.png')
        plt.show()
        
        print("Visualization saved as 'visualizations/simulation_history.png'")
        
        # Additional visualization for seasonal or spatial simulations
        if self.simulation_type in ['seasonal', 'spatial']:
            plt.figure(figsize=(10, 8))
            
            if self.simulation_type == 'seasonal':
                # Create heatmap of feature values vs. predictions
                pivot_df = history_df.copy()
                pivot_df['prediction'] = pivot_df['prediction'].astype(str)
                
                # Create a correlation matrix
                corr = history_df.drop(columns=['timestamp', 'prediction']).corr()
                
                # Plot the correlation heatmap
                sns.heatmap(corr, annot=True, cmap='coolwarm')
                plt.title('Feature Correlation in Seasonal Simulation')
                plt.tight_layout()
                plt.savefig('visualizations/seasonal_correlation.png')
                plt.show()
                
            elif self.simulation_type == 'spatial':
                # Plot the spatial movement and predictions
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(
                    history_df['NDVI_N'], 
                    history_df['Water_N'],
                    c=history_df['prediction'],
                    cmap='coolwarm',
                    s=50,
                    alpha=0.7
                )
                
                plt.colorbar(scatter, label='Conflict Risk (1=High, 0=Low)')
                plt.xlabel('NDVI_N')
                plt.ylabel('Water_N')
                plt.title('Spatial Distribution of Conflict Risk')
                plt.grid(True)
                plt.savefig('visualizations/spatial_distribution.png')
                plt.show()
                
                print(f"Additional visualization saved for {self.simulation_type} simulation.")
    
    def save_history(self, filename='simulation_data.csv'):
        """Save the simulation history to a CSV file"""
        if not self.data_history:
            print("No simulation data available to save.")
            return
        
        # Convert histories to DataFrame
        history_df = pd.DataFrame(self.data_history)
        history_df['timestamp'] = self.timestamp_history
        history_df['prediction'] = self.prediction_history
        
        # Save to CSV
        history_df.to_csv(f'model_outputs/{filename}', index=False)
        print(f"Simulation data saved to 'model_outputs/{filename}'")


def main():
    """Main function to run the IoT simulator"""
    # Set up the simulator
    simulator = IoTSimulator(simulation_type='spatial')
    
    # Run the simulation
    simulator.run_simulation(iterations=50, delay=0.1)
    
    # Visualize the results
    simulator.visualize_history()
    
    # Save the results
    simulator.save_history()

if __name__ == "__main__":
    main()
```

# 3. data_preprocessing.py
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for processed data
os.makedirs('processed_data', exist_ok=True)

def load_and_explore_data(filepath):
    """Load and explore the dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check data types
    print("\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Return dataframe
    return df

def preprocess_data(df, normalize_columns=None, scale_method='minmax'):
    """Preprocess the dataset"""
    print("\nPreprocessing data...")
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    if processed_df.isnull().sum().sum() > 0:
        print("Handling missing values...")
        
        # For numerical columns, fill with median
        num_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            if processed_df[col].isnull().sum() > 0:
                processed_df[col].fillna(processed_df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        cat_cols = processed_df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if processed_df[col].isnull().sum() > 0:
                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
    
    # Normalize specified columns
    if normalize_columns:
        print(f"Normalizing columns using {scale_method} method...")
        
        # Check if all columns exist
        for col in normalize_columns:
            if col not in processed_df.columns:
                print(f"Warning: Column {col} not found in dataset.")
                normalize_columns.remove(col)
        
        # Create a copy of the original values
        for col in normalize_columns:
            processed_df[f"{col}_original"] = processed_df[col]
        
        # Apply scaling
        if scale_method == 'minmax':
            scaler = MinMaxScaler()
        else:  # 'standard'
            scaler = StandardScaler()
        
        processed_df[normalize_columns] = scaler.fit_transform(processed_df[normalize_columns])
        
        # Add suffix to normalized columns
        for col in normalize_columns:
            processed_df.rename(columns={col: f"{col}_N"}, inplace=True)
            # Move original values back
            processed_df[col] = processed_df[f"{col}_original"]
            processed_df.drop(columns=[f"{col}_original"], inplace=True)
    
    return processed_df

def calculate_habitat_suitability(df, weights=None):
    """Calculate Habitat Suitability Index using weighted sum model"""
    print("\nCalculating Habitat Suitability Index...")
    
    # Default weights if none provided
    if weights is None:
        weights = {
            'NDVI_N': 0.3,
            'Water_N': 0.25,
            'Slope_N': 0.15, 
            'Land_N': 0.1,
            'Conflict_R_N': 0.2
        }
    
    # Check if all weighted columns exist
    for col in weights.keys():
        if col not in df.columns:
            print(f"Warning: Column {col} not found in dataset. Skipping in HSI calculation.")
            weights.pop(col)
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Compute WSM-based HSI
    df['HSI_WSM'] = sum(df[var] * w for var, w in weights.items())
    
    print("HSI calculation complete.")
    print(f"Weights used: {weights}")
    
    return df

def generate_synthetic_conflict_data(df):
    """Generate synthetic conflict data for simulation"""
    print("\nGenerating synthetic conflict data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define possible values for simulation
    conflict_species = ['Elephant', 'Leopard', 'Wild Boar', 'Tiger', 'Rhino']
    conflict_types = ['Crop Raiding', 'Human Attack', 'Livestock Killing', 'Property Damage']
    villages = ['Village_A', 'Village_B', 'Village_C', 'Village_D']
    years = list(range(2018, 2024))
    
    # Species distribution (more elephants and wild boars)
    species_probs = [0.4, 0.2, 0.3, 0.05, 0.05]
    
    # Types distribution (more crop raiding)
    types_probs = [0.6, 0.1, 0.2, 0.1]
    
    # Add simulated columns to the DataFrame
    enhanced_df = df.copy()
    enhanced_df['Conflict_Species'] = np.random.choice(conflict_species, size=len(enhanced_df), p=species_probs)
    enhanced_df['Conflict_Type'] = np.random.choice(conflict_types, size=len(enhanced_df), p=types_probs)
    enhanced_df['Year'] = np.random.choice(years, size=len(enhanced_df))
    enhanced_df['Village'] = np.random.choice(villages, size=len(enhanced_df))
    
    # If HSI_WSM exists, make conflict species related to HSI
    if 'HSI_WSM' in enhanced_df.columns:
        # Higher HSI areas more likely to have elephants and tigers
        high_hsi_mask = enhanced_df['HSI_WSM'] > enhanced_df['HSI_WSM'].median()
        # Randomly select rows to modify (80% of high HSI areas)
        modify_mask = high_hsi_mask & (np.random.random(size=len(enhanced_df)) < 0.8)
        # Assign elephants or tigers to these areas
        enhanced_df.loc[modify_mask, 'Conflict_Species'] = np.random.choice(
            ['Elephant', 'Tiger'], size=modify_mask.sum(), p=[0.8, 0.2]
        )
    
    print("Synthetic conflict data generated.")
    print(enhanced_df[['Conflict_Species', 'Conflict_Type', 'Year', 'Village']].value_counts().head())
    
    return enhanced_df

def visualize_data(df):
    """Create visualizations of the dataset"""
    print("\nCreating data visualizations...")
    
    # Create directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Correlation Heatmap
    if any(col.endswith('_N') for col in df.columns):
        # Select only numerical columns for correlation
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_df = df[num_cols]
        
        # Remove columns with all same value
        numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
        
        plt.figure(figsize=(12, 10))
        correlation = numeric_df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png')
        print("Correlation heatmap saved.")
    
    # 2. WSM Heatmap (if spatial data is available)
    if all(col in df.columns for col in ['row_index', 'col_index']) and 'HSI_WSM' in df.columns:
        plt.figure(figsize=(12, 10))
        heatmap_data = df.pivot_table(index='row_index', columns='col_index', values='HSI_WSM', aggfunc='mean')
        
        # Create heatmap with grid lines
        ax = sns.heatmap(heatmap_data, cmap='YlGnBu', cbar_kws={'label': 'HSI_WSM'})
        
        # Add grid lines
        ax.set_xticks(np.arange(heatmap_data.shape[1]+1)-0.5, minor=True)
        ax.set_yticks(np.arange(heatmap_data.shape[0]+1)-0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        
        plt.title('Habitat Suitability Heatmap')
        plt.tight_layout()
        plt.savefig('visualizations/hsi_heatmap.png')
        print("HSI heatmap saved.")
    
    # 3. Conflict Risk Heatmap
    if all(col in df.columns for col in ['row_index', 'col_index']) and 'Conflict_R' in df.columns:
        plt.figure(figsize=(12, 10))
        conflict_data = df.pivot_table(index='row_index', columns='col_index', values='Conflict_R', aggfunc='mean')
        
        # Create heatmap with grid lines
        ax = sns.heatmap(conflict_data, cmap='Reds', cbar_kws={'label': 'Conflict Risk'})
        
        # Add grid lines
        ax.set_xticks(np.arange(conflict_data.shape[1]+1)-0.5, minor=True)
        ax.set_yticks(np.arange(conflict_data.shape[0]+1)-0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        
        plt.title('Conflict Risk Heatmap')
        plt.tight_layout()
        plt.savefig('visualizations/conflict_risk_heatmap.png')
        print("Conflict risk heatmap saved.")
    
    # 4. Species distribution
    if 'Conflict_Species' in df.columns:
        plt.figure(figsize=(12, 6))
        species_counts = df['Conflict_Species'].value_counts()
        species_counts.plot(kind='bar', color='skyblue')
        plt.title('Conflict Species Distribution')
        plt.xlabel('Species')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/species_distribution.png')
        print("Species distribution visualization saved.")
    
    # 5. NDVI vs Slope scatter plot with conflict risk color
    if all(col in df.columns for col in ['NDVI_N', 'Slope_N', 'Conflict_R']):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            df['NDVI_N'], 
            df['Slope_N'],
            c=df['Conflict_R'],
            cmap='coolwarm',
            alpha=0.7,
            s=50
        )
        plt.colorbar(scatter, label='Conflict Risk')
        plt.xlabel('NDVI_N')
        plt.ylabel('Slope_N')
        plt.title('NDVI vs Slope with Conflict Risk')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('visualizations/ndvi_slope_scatter.png')
        print("NDVI vs Slope scatter plot saved.")
    
    print("Data visualization complete.")

def main():
    """Main function for data preprocessing pipeline"""
    # Load and explore data
    df = load_and_explore_data("improved_habitat_suitability.csv")
    
    # Preprocess data
    normalize_columns = ['NDVI', 'Water_Proximity', 'Slope', 'Land_Recovery', 'Conflict_R']
    processed_df = preprocess_data(df, normalize_columns=normalize_columns)
    
    # Calculate habitat suitability
    hsi_df = calculate_habitat_suitability(processed_df)
    
    # Generate synthetic conflict data
    enhanced_df = generate_synthetic_conflict_data(hsi_df)
    
    # Visualize data
    visualize_data(enhanced_df)
    
    # Save processed data
    enhanced_df.to_csv('processed_data/enhanced_wildlife_data.csv', index=False)
    print("\nProcessed data saved to 'processed_data/enhanced_wildlife_data.csv'")

if __name__ == "__main__":
    main()
```

# 4. visualization.py
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Create directory for visualizations
os.makedirs('visualizations', exist_ok=True)

def create_heatmaps(df, with_grid=True):
    """Create heatmaps for HSI and conflict risk with grid lines"""
    print("Creating heatmaps...")
    
    # Check if spatial data is available
    if not all(col in df.columns for col in ['row_index', 'col_index']):
        print("Warning: Spatial data (row_index, col_index) not found. Cannot create heatmaps.")
        return
    
    # Create HSI heatmap if available
    if 'HSI_WSM' in df.columns:
        hsi_data = df.pivot_table(index='row_index', columns='col_index', values='HSI_WSM', aggfunc='mean')
        
        plt.figure(figsize=(14, 12))
        ax = sns.heatmap(
            hsi_data, 
            cmap='YlGnBu', 
            cbar_kws={'label': 'Habitat Suitability Index'}
        )
        
        # Add grid lines if requested
        if with_grid:
            # Add grid lines
            ax.set_xticks(np.arange(hsi_data.shape[1]+1)-0.5, minor=True)
            ax.set_yticks(np.arange(hsi_data.shape[0]+1)-0.5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)
        
        plt.title('Habitat Suitability Index (HSI) Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig('visualizations/hsi_heatmap_with_grid.png')
        print("HSI heatmap saved.")
    
    # Create conflict risk heatmap if available
    if 'Conflict_R' in df.columns:
        conflict_data = df.pivot_table(index='row_index', columns='col_index', values='Conflict_R', aggfunc='mean')
        
        plt.figure(figsize=(14, 12))
        ax = sns.heatmap(
            conflict_data, 
            cmap='Reds', 
            cbar_kws={'label': 'Conflict Risk'}
        )
        
        # Add grid lines if requested
        if with_grid:
            # Add grid lines
            ax.set_xticks(np.arange(conflict_data.shape[1]+1)-0.5, minor=True)
            ax.set_yticks(np.arange(conflict_data.shape[0]+1)-0.5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)
        
        plt.title('Conflict Risk Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig('visualizations/conflict_risk_heatmap_with_grid.png')
        print("Conflict risk heatmap saved.")
    
    # Create interactive plotly heatmaps for better visualization
    if 'HSI_WSM' in df.columns:
        fig = px.imshow(
            hsi_data,
            color_continuous_scale='Viridis',
            labels=dict(color="HSI_WSM"),
            title="Interactive Habitat Suitability Index Heatmap"
        )
        
        # Add grid lines if requested
        if with_grid:
            # Add horizontal grid lines
            for i in range(hsi_data.shape[0] + 1):
                fig.add_shape(
                    type='line',
                    x0=-0.5, y0=i-0.5, x1=hsi_data.shape[1]-0.5, y1=i-0.5,
                    line=dict(color='White', width=1)
                )
            
            # Add vertical grid lines
            for i in range(hsi_data.shape[1] + 1):
                fig.add_shape(
                    type='line',
                    x0=i-0.5, y0=-0.5, x1=i-0.5, y1=hsi_data.shape[0]-0.5,
                    line=dict(color='White', width=1)
                )
        
        fig.update_layout(width=900, height=700)
        fig.write_html('visualizations/interactive_hsi_heatmap.html')
        print("Interactive HSI heatmap saved.")

def create_species_analysis(df):
    """Create visualizations for species-specific analysis"""
    print("Creating species analysis visualizations...")
    
    if 'Conflict_Species' not in df.columns:
        print("Warning: Conflict_Species column not found. Cannot create species analysis.")
        return
    
    # Species distribution
    plt.figure(figsize=(12, 6))
    species_counts = df['Conflict_Species'].value_counts()
    ax = species_counts.plot(kind='bar', color='skyblue')
    
    # Add count labels on top of bars
    for i, count in enumerate(species_counts):
        ax.text(i, count + 0.1, str(count), ha='center', fontsize=10)
    
    plt.title('Conflict Species Distribution', fontsize=14)
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/species_distribution.png')
    
    # Species by conflict type
    if 'Conflict_Type' in df.columns:
        plt.figure(figsize=(14, 8))
        species_type_counts = pd.crosstab(df['Conflict_Species'], df['Conflict_Type'])
        species_type_counts.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Conflict Types by Species', fontsize=14)
        plt.xlabel('Species')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Conflict Type')
        plt.tight_layout()
        plt.savefig('visualizations/species_by_conflict_type.png')
    
    # Species by year if available
    if 'Year' in df.columns:
        plt.figure(figsize=(14, 8))
        species_year_counts = pd.crosstab(df['Year'], df['Conflict_Species'])
        species_year_counts.plot(kind='line', marker='o', colormap='tab10')
        plt.title('Species Conflicts Over Time', fontsize=14)
        plt.xlabel('Year')
        plt.ylabel('Number of Conflicts')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Species')
        plt.tight_layout()
        plt.savefig('visualizations/species_over_time.png')
    
    # Create an interactive dashboard for species analysis
    if all(col in df.columns for col in ['Conflict_Species', 'Conflict_Type', 'HSI_WSM']):
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Species Distribution', 
                'Conflict Types by Species',
                'HSI Distribution by Species',
                'Conflict Locations'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Species Distribution
        species_counts = df['Conflict_Species'].value_counts().reset_index()
        species_counts.columns = ['Species', 'Count']
        
        fig.add_trace(
            go.Bar(
                x=species_counts['Species'],
                y=species_counts['Count'],
                name='Species Count',
                marker_color='skyblue'
            ),
            row=1, col=1
        )
        
        # 2. Conflict Types by Species
        for conflict_type in df['Conflict_Type'].unique():
            type_counts = df[df['Conflict_Type'] == conflict_type]['Conflict_Species'].value_counts().reset_index()
            type_counts.columns = ['Species', 'Count']
            
            fig.add_trace(
                go.Bar(
                    x=type_counts['Species'],
                    y=type_counts['Count'],
                    name=conflict_type
                ),
                row=1, col=2
            )
        
        # 3. HSI Distribution by Species
        for species in df['Conflict_Species'].unique():
            species_data = df[df['Conflict_Species'] == species]
            
            fig.add_trace(
                go.Box(
                    y=species_data['HSI_WSM'],
                    name=species,
                    boxmean=True
                ),
                row=2, col=1
            )
        
        # 4. Conflict Locations (if spatial data available)
        if all(col in df.columns for col in ['row_index', 'col_index']):
            for species in df['Conflict_Species'].unique():
                species_data = df[df['Conflict_Species'] == species]
                
                fig.add_trace(
                    go.Scatter(
                        x=species_data['row_index'],
                        y=species_data['col_index'],
                        mode='markers',
                        name=species,
                        marker=dict(
                            size=8,
                            opacity=0.6
                        )
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Species Analysis Dashboard",
            height=800,
            width=1200,
            barmode='stack',
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Species", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        fig.update_xaxes(title_text="Species", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_xaxes(title_text="Species", row=2, col=1)
        fig.update_yaxes(title_text="HSI_WSM", row=2, col=1)
        
        fig.update_xaxes(title_text="Row Index", row=2, col=2)
        fig.update_yaxes(title_text="Column Index", row=2, col=2)
        
        # Save the figure
        fig.write_html('visualizations/interactive_species_dashboard.html')
        print("Interactive species dashboard saved.")

def create_feature_importance_visualization(model_path='model_outputs/conflict_risk_model.pkl'):
    """Create visualization for feature importance if model is available"""
    try:
        import joblib
        model = joblib.load(model_path)
        
        if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
            feature_importance = model.feature_importances_
            feature_names = model.feature_names_in_
            
            # Sort features by importance
            indices = np.argsort(feature_importance)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importance", fontsize=16)
            plt.bar(range(len(indices)), feature_importance[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png')
            print("Feature importance visualization saved.")
            
            # Interactive visualization
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                title='Feature Importance',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                xaxis_title='Feature',
                yaxis_title='Importance',
                coloraxis_showscale=False
            )
            
            fig.write_html('visualizations/interactive_feature_importance.html')
            print("Interactive feature importance visualization saved.")
    
    except Exception as e:
        print(f"Could not create feature importance visualization: {str(e)}")

def main():
    """Main function for creating visualizations"""
    try:
        # Load data
        df = pd.read_csv('processed_data/enhanced_wildlife_data.csv')
        
        # Create heatmaps with grid lines
        create_heatmaps(df, with_grid=True)
        
        # Create species analysis
        create_species_analysis(df)
        
        # Create feature importance visualization
        create_feature_importance_visualization()
        
        print("All visualizations created successfully.")
    
    except Exception as e:
        print(f"Error in visualization creation: {str(e)}")

if __name__ == "__main__":
    main()
```

# 5. dashboard.py
```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import time
from datetime import datetime
import os

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model_path = 'model_outputs/conflict_risk_model.pkl'
    scaler_path = 'model_outputs/feature_scaler.pkl'
    
    model = joblib.load(model_path)
    
    try:
        scaler = joblib.load(scaler_path)
        use_scaler = True
    except:
        scaler = None
        use_scaler = False
    
    return model, scaler, use_scaler

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('processed_data/enhanced_wildlife_data.csv')

# Define the Streamlit app
def main():
    # Page config
    st.set_page_config(
        page_title="Wildlife Conflict Risk Dashboard",
        page_icon="🦁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load model, scaler and data
    model, scaler, use_scaler = load_model_and_scaler()
    df = load_data()
    
    # Get model features
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_
    else:
        features = ['NDVI_N', 'Slope_N', 'Water_N', 'Land_N']
    
    # Sidebar
    with st.sidebar:
        st.title("🦁 Wildlife Conflict Dashboard")
        st.markdown("### Filters & Controls")
        
        # Filters
        st.subheader("Data Filters")
        
        # Species filter if available
        if 'Conflict_Species' in df.columns:
            species_filter = st.selectbox(
                "Select Species",
                options=["All"] + sorted(df['Conflict_Species'].unique().tolist()),
                index=0
            )
        else:
            species_filter = "All"
        
        # Village filter if available
        if 'Village' in df.columns:
            village_filter = st.selectbox(
                "Select Village",
                options=["All"] + sorted(df['Village'].unique().tolist()),
                index=0
            )
        else:
            village_filter = "All"
        
        # Year filter if available
        if 'Year' in df.columns:
            min_year = int(df['Year'].min())
            max_year = int(df['Year'].max())
            year_filter = st.slider(
                "Select Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        else:
            year_filter = None
        
        # Simulation controls
        st.subheader("Simulation")
        simulation_type = st.selectbox(
            "Simulation Type",
            options=["Random", "Seasonal", "Spatial"],
            index=0
        )
        
        update_interval = st.slider(
            "Update Interval (seconds)",
            min_value=1,
            max_value=10,
            value=3
        )
        
        start_simulation = st.button("Start/Stop Simulation")
    
    # Filter data based on selections
    filtered_df = df.copy()
    
    if species_filter != "All" and 'Conflict_Species' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Conflict_Species'] == species_filter]
    
    if village_filter != "All" and 'Village' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Village'] == village_filter]
    
    if year_filter and 'Year' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['Year'] >= year_filter[0]) & 
                                 (filtered_df['Year'] <= year_filter[1])]
    
    # Main page content
    st.title("Wildlife Habitat Suitability & Conflict Risk Dashboard")
    
    # Show active filters
    filter_msg = f"Filters: "
    filter_msg += f"Species: {species_filter}, " if 'Conflict_Species' in df.columns else ""
    filter_msg += f"Village: {village_filter}, " if 'Village' in df.columns else ""
    filter_msg += f"Years: {year_filter[0]}-{year_filter[1]}" if year_filter else ""
    st.markdown(f"*{filter_msg}*")
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Habitat & Risk Maps", 
        "Species Analysis", 
        "Feature Analysis",
        "Real-time Simulation"
    ])
    
    with tab1:
        st.header("Habitat and Risk Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Habitat Suitability Heatmap
            if all(col in filtered_df.columns for col in ['row_index', 'col_index', 'HSI_WSM']):
                st.subheader("Habitat Suitability Index (HSI) Heatmap")
                
                hsi_data = filtered_df.pivot_table(
                    index='row_index', 
                    columns='col_index', 
                    values='HSI_WSM', 
                    aggfunc='mean'
                )
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(hsi_data, annot=False, cmap="YlGnBu", ax=ax)
                
                # Add grid
                ax.set_xticks(np.arange(hsi_data.shape[1]+1)-0.5, minor=True)
                ax.set_yticks(np.arange(hsi_data.shape[0]+1)-0.5, minor=True)
                ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
                ax.tick_params(which="minor", bottom=False, left=False)
                
                st.pyplot(fig)
            else:
                st.warning("Habitat Suitability data not available with current filters.")
        
        with col2:
            # Conflict Risk Heatmap
            if all(col in filtered_df.columns for col in ['row_index', 'col_index', 'Conflict_R']):
                st.subheader("Conflict Risk Heatmap")
                
                conflict_data = filtered_df.pivot_table(
                    index='row_index', 
                    columns='col_index', 
                    values='Conflict_R', 
                    aggfunc='mean'
                )
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(conflict_data, annot=False, cmap="Reds", ax=ax)
                
                # Add grid
                ax.set_xticks(np.arange(conflict_data.shape[1]+1)-0.5, minor=True)
                ax.set_yticks(np.arange(conflict_data.shape[0]+1)-0.5, minor=True)
                ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
                ax.tick_params(which="minor", bottom=False, left=False)
                
                st.pyplot(fig)
            else:
                st.warning("Conflict Risk data not available with current filters.")
        
        # Interactive 3D Surface Plot
        if all(col in filtered_df.columns for col in ['row_index', 'col_index', 'HSI_WSM']):
            st.subheader("3D Surface Plot of Habitat Suitability")
            
            hsi_data = filtered_df.pivot_table(
                index='row_index', 
                columns='col_index', 
                values='HSI_WSM', 
                aggfunc='mean'
            )
            
            fig = go.Figure(data=[go.Surface(z=hsi_data.values)])
            fig.update_layout(
                title='3D Habitat Suitability Surface',
                scene=dict(
                    xaxis_title='Column Index',
                    yaxis_title='Row Index',
                    zaxis_title='HSI Value'
                ),
                width=800,
                height=700
            )
            
            st.plotly_chart(fig)
    
    with tab2:
        st.header("Species Analysis")
        
        if 'Conflict_Species' in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Species distribution pie chart
                st.subheader("Conflict Species Distribution")
                species_counts = filtered_df['Conflict_Species'].value_counts().reset_index()
                species_counts.columns = ['Species', 'Count']
                
                fig = px.pie(
                    species_counts, 
                    values='Count', 
                    names='Species',
                    title='Conflict Species Distribution',
                    hole=0.4
                )
                
                st.plotly_chart(fig)
            
            with col2:
                # Species by conflict type
                if 'Conflict_Type' in filtered_df.columns:
                    st.subheader("Conflict Types by Species")
                    
                    conflict_type_counts = pd.crosstab(
                        filtered_df['Conflict_Species'], 
                        filtered_df['Conflict_Type']
                    )
                    fig = px.bar(
                        conflict_type_counts, 
                        x=conflict_type_counts.index, 
                        y=conflict_type_counts.columns,
                        title='Conflict Types by Species',
                        labels={'value': 'Count', 'index': 'Species'}
                    )