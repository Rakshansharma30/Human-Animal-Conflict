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
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}.  Please make sure the file exists and the path is correct.")
        return None  # Return None on error
    
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
                processed_df[col] = processed_df[col].fillna(processed_df[col].median()) # Changed to direct assignment
        
        # For categorical columns, fill with mode
        cat_cols = processed_df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if processed_df[col].isnull().sum() > 0:
                processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0]) # Changed to direct assignment
    
    # Normalize specified columns
    if normalize_columns:
        print(f"Normalizing columns using {scale_method} method...")
        
        # Check if all columns exist
        for col in normalize_columns:
            if col not in processed_df.columns:
                print(f"Warning: Column {col} not found in dataset. Skipping in normalization.")
                normalize_columns = [c for c in normalize_columns if c in processed_df.columns] #remove invalid cols
                if not normalize_columns:
                    print("No valid columns to normalize. Skipping normalization step.")
                    return processed_df
                break # Stop checking if one is not found
        
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
            processed_df = processed_df.rename(columns={col: f"{col}_N"}) # Changed to direct assignment
             # Move original values back
            processed_df[col] = processed_df[f"{col}_original"]
            processed_df = processed_df.drop(columns=[f"{col}_original"]) # Changed to direct assignment
            
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
            weights = {k: v for k, v in weights.items() if k in df.columns} # Filter weights
            if not weights:
                print("No valid weights.  Returning original DataFrame.")
                return df
            break
            
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Compute WSM-based HSI
    # Check for duplicate columns before calculation
    if len(df.columns) != len(set(df.columns)):
        print("Error: DataFrame contains duplicate column names.  Please ensure column names are unique before calculating HSI.")
        return df.copy()  # Return a copy to avoid modifying the original DataFrame
    
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
    
    # 5. NDVI vs Slope scatter plot with conflict risk color - FIXED VERSION
    try:
        plt.figure(figsize=(10, 8))
        
        # Check which columns are available in numeric format
        if 'NDVI_N' in df.columns and 'Slope_N' in df.columns:
            # Create a simpler scatter plot without using the conflict risk as color
            # We'll use a generic color instead
            plt.scatter(
                df['NDVI_N'].values, 
                df['Slope_N'].values,
                c='blue',      # Use a fixed color instead
                alpha=0.5,
                s=20
            )
            plt.xlabel('Normalized NDVI')
            plt.ylabel('Normalized Slope')
            plt.title('NDVI vs Slope')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('visualizations/ndvi_slope_scatter.png')
            print("NDVI vs Slope scatter plot saved.")
        else:
            print("Required columns for scatter plot not found. Skipping.")
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        print("Skipping scatter plot visualization.")
    
    print("Data visualization complete.")

def main():
    """Main function for data preprocessing pipeline"""
    # Load and explore data
    filepath = "wsm_output.csv"  # <--- CHANGE THIS LINE TO YOUR FILE NAME
    df = load_and_explore_data(filepath)
    
    if df is None:
        print("Error: Data loading failed.  Exiting.")
        return
    
    # Preprocess data
    # Adjust these column names to match your "wsm_output.csv" file
    normalize_columns = ['NDVI', 'Water_Proximity', 'Slope', 'Land_Recovery', 'Conflict_R']  # Example
    processed_df = preprocess_data(df, normalize_columns=normalize_columns)
    
    # Calculate habitat suitability
    hsi_df = calculate_habitat_suitability(processed_df)
    
    if hsi_df is None:
        print("Error: HSI calculation failed.  Exiting.")
        return
    
    # Generate synthetic conflict data
    enhanced_df = generate_synthetic_conflict_data(hsi_df)
    
    # Visualize data
    visualize_data(enhanced_df)
    
    # Save processed data
    enhanced_df.to_csv('processed_data/enhanced_wildlife_data.csv', index=False)
    print("\nProcessed data saved to 'processed_data/enhanced_wildlife_data.csv'")

if __name__ == "__main__":
    main()