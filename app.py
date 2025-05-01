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
                    ).reset_index()
                    
                    # Melt the dataframe for plotting
                    melted_df = conflict_type_counts.melt(id_vars='Conflict_Species', var_name='Conflict_Type', value_name='Count')
                    
                    fig = px.bar(
                        melted_df, 
                        x='Conflict_Species', 
                        y='Count',
                        color='Conflict_Type',
                        title='Conflict Types by Species',
                        labels={'Conflict_Species': 'Species'}
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning("Conflict Type data is not available.")
        else:
            st.warning("Conflict Species data is not available.")
    
    with tab3:
        st.header("Feature Analysis")
        
        # Feature importance plot
        if hasattr(model, 'feature_importances_') and use_scaler:
            st.subheader("Feature Importance (Scaled Data)")
            
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='Feature',
                y='Importance',
                title='Feature Importance (Scaled Data)'
            )
            st.plotly_chart(fig)
        elif hasattr(model, 'feature_importances_') and not use_scaler:
             st.subheader("Feature Importance (Unscaled Data)")
            
             importances = model.feature_importances_
             feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
             feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
             fig = px.bar(
                feature_importance,
                x='Feature',
                y='Importance',
                title='Feature Importance (Unscaled Data)'
             )
             st.plotly_chart(fig)
        else:
            st.warning("Model does not support feature importance.")
        
        # Feature distribution
        st.subheader("Feature Distributions")
        
        cols = st.columns(len(features))
        for i, feature in enumerate(features):
            with cols[i]:
                fig, ax = plt.subplots()
                sns.histplot(filtered_df[feature], kde=True, ax=ax)
                ax.set_title(feature)
                st.pyplot(fig)
    
    with tab4:
        st.header("Real-time Simulation")
        
        if 'row_index' in df.columns and 'col_index' in df.columns:
            conflict_risk_placeholder = st.empty()
            
            # Simulate data and update plot
            for _ in range(100):  # Run for a fixed number of iterations
                
                if simulation_type == "Random":
                    df['Conflict_R'] = np.random.uniform(0, 1, size=len(df))
                elif simulation_type == "Seasonal":
                    # Simulate higher conflict in certain months (e.g., months 6-9)
                    month = datetime.now().month
                    if 6 <= month <= 9:
                        df['Conflict_R'] = np.random.uniform(0.7, 1, size=len(df))  # High conflict
                    else:
                        df['Conflict_R'] = np.random.uniform(0, 0.3, size=len(df)) # Low conflict
                elif simulation_type == "Spatial":
                    # Create a spatial pattern (e.g., higher conflict near forest edges)
                    df['Conflict_R'] = np.clip(0.2 + 0.8 * np.exp(-((df['row_index'] - df['row_index'].mean())**2 + (df['col_index'] - df['col_index'].mean())**2) / 100),0,1)
                
                
                # Filter data
                sim_filtered_df = df.copy()
                if species_filter != "All" and 'Conflict_Species' in sim_filtered_df.columns:
                    sim_filtered_df = sim_filtered_df[sim_filtered_df['Conflict_Species'] == species_filter]
                if village_filter != "All" and 'Village' in sim_filtered_df.columns:
                    sim_filtered_df = sim_filtered_df[sim_filtered_df['Village'] == village_filter]
                if year_filter and 'Year' in sim_filtered_df.columns:
                    sim_filtered_df = sim_filtered_df[(sim_filtered_df['Year'] >= year_filter[0]) & 
                                                        (sim_filtered_df['Year'] <= year_filter[1])]
                
                
                conflict_data = sim_filtered_df.pivot_table(
                    index='row_index', 
                    columns='col_index', 
                    values='Conflict_R', 
                    aggfunc='mean'
                )
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(conflict_data, annot=False, cmap="Reds", ax=ax, vmin=0, vmax=1) # Fix the colorbar scale
                ax.set_title(f'Simulated Conflict Risk (Iteration {_})')
                
                # Add grid
                ax.set_xticks(np.arange(conflict_data.shape[1]+1)-0.5, minor=True)
                ax.set_yticks(np.arange(conflict_data.shape[0]+1)-0.5, minor=True)
                ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
                ax.tick_params(which="minor", bottom=False, left=False)
                
                conflict_risk_placeholder.pyplot(fig)
                time.sleep(update_interval)
        else:
            st.warning("row_index and col_index are needed for Real-time Simulation")

if __name__ == "__main__":
    main()
