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