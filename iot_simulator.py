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
