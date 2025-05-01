# Human-Wildlife Conflict Analysis and Prediction

## Project Overview

This project aims to analyze and predict human-wildlife conflict (HWC) using a combination of data from government reports, scientific literature, NGO databases, and community-level reports. The methodology involves grid mapping, conflict scoring, data preprocessing, and machine learning for risk prediction.

## Methodology

### 5.1 Data Collection

* Government reports and HWC databases from forest departments in India and Nepal
* Scientific literature and case studies
* Data shared by conservation NGOs like WWF, NTNC, and WII
* Community-level conflict reports and spatial GPS data

Each incident was geotagged (where possible) and recorded with metadata, including species, time, severity, and type (e.g., crop damage, livestock loss, human injury).

### 5.2 Grid Mapping and Cell Division

Each national park or reserve was segmented into a grid of uniform cells. A grid cell was assigned coordinates (i, j) with a conflict score derived from incident frequency and severity.

### 5.3 Conflict Scoring System

A scoring rubric was applied to standardize data:

* 0: No incidents
* 1: Minor property/crop loss
* 2: Livestock loss
* 3: Human injury or recurring property damage
* 4: Human fatality or frequent multi-species incidents

### 5.4 Data Preprocessing

Steps involved:

* Converting raw reports into structured tabular format (CSV)
* Cleaning missing data and resolving spatial inaccuracies
* Normalizing date and time fields
* Binning locations into grid coordinates using spatial transformation

### 5.5 Visualization Tools
*(Details on visualization tools used would be added here)*

### 5.6 Algorithm Design

A machine learning approach was employed to predict HWC risk.

1.  **Feature Selection:** Relevant features were selected, including:
    * Habitat Suitability Index (HSI)
    * Distance to protected area boundaries
    * Human population density
    * Livestock density
    * Land use patterns
    * Elevation and slope
    * Proximity to water sources
    * Historical conflict data

2.  **Model Selection:** Machine learning algorithms considered:
    * Logistic Regression
    * Random Forest
    * Gradient Boosting Machines (e.g., XGBoost, LightGBM)

    The final model was selected based on performance metrics and suitability for the data.

3.  **Model Training and Validation:** The dataset was divided into training, validation, and testing sets for model training, hyperparameter tuning, and performance evaluation.

4.  **Conflict Risk Prediction:** The trained model was used to predict the probability of conflict occurrence for each grid cell, which was then translated into a conflict risk level.

### 5.7 Model Evaluation

The performance of the conflict risk prediction model was evaluated using metrics such as:

* AUC-ROC
* Precision and Recall
* F1-score
* Confusion Matrix

### 5.8 System Implementation

The methodology was implemented using Python libraries, including:

* Pandas: For data manipulation and preprocessing.
* Scikit-learn: For machine learning.
* Matplotlib and Plotly: For data visualization.
* Streamlit: For creating the interactive web application.

The system comprises two main components:

1.  Data Processing Module (`Data Processing.py`): Performs data handling, HSI calculation, and synthetic data generation.
2.  Interactive Dashboard (`wildlife_conflict_dashboard.py`): Provides a user interface for data exploration, risk visualization, and simulations.

### 5.9 Limitations

* Data quality and availability
* Use of synthetic conflict data
* Model assumptions regarding species-habitat relationships and conflict drivers

### 5.10 Future Research

* Improving data collection methods
* Refining the HSI model
* Exploring advanced machine learning techniques
* Conducting field validation of model predictions
