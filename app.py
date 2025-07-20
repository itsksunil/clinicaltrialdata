import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

# App title and description
st.title("ClinicalTrials.gov Advanced Analytics")
st.write("""
This app queries clinical trials data from ClinicalTrials.gov API with custom filters, 
provides visualizations, and offers basic neural network analysis capabilities.
""")

# Sidebar for filters
st.sidebar.header("Filter Options")

# Example filter options
condition = st.sidebar.text_input("Medical Condition (e.g., diabetes)")
intervention = st.sidebar.text_input("Intervention (e.g., drug name)")
country = st.sidebar.text_input("Country")
study_type = st.sidebar.selectbox("Study Type", ["", "Interventional", "Observational", "Expanded Access"])
phase = st.sidebar.selectbox("Phase", ["", "Phase 1", "Phase 2", "Phase 3", "Phase 4"])
status = st.sidebar.selectbox("Status", ["", "Recruiting", "Completed", "Terminated"])

# Date range filter
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime(2010, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime.today())

# Main search button
if st.button("Search Clinical Trials"):
    # Build query parameters based on filters
    params = {}
    
    if condition:
        params["condition"] = condition
    if intervention:
        params["intervention"] = intervention
    if country:
        params["country"] = country
    if study_type:
        params["studyType"] = study_type
    if phase:
        params["phase"] = phase
    if status:
        params["status"] = status
    
    # Convert dates to string format
    params["fromDate"] = start_date.strftime("%m/%d/%Y")
    params["toDate"] = end_date.strftime("%m/%d/%Y")
    
    # Make API request
    try:
        st.write("Searching for trials with the following criteria:")
        st.json(params)
        
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Display basic info
        total_studies = data.get('totalStudies', 0)
        st.subheader(f"Found {total_studies} trials")
        
        if total_studies == 0:
            st.warning("No trials found with these filters.")
            st.stop()
        
        # Convert to DataFrame for analysis
        studies = data.get('studies', [])
        df_data = []
        
        for study in studies:
            protocol = study.get('protocolSection', {})
            ident = protocol.get('identificationModule', {})
            status_module = protocol.get('statusModule', {})
            design = protocol.get('designModule', {})
            
            df_data.append({
                'nctId': ident.get('nctId'),
                'title': ident.get('briefTitle'),
                'condition': ', '.join(ident.get('conditions', [])),
                'intervention': ', '.join([i.get('name', '') for i in protocol.get('interventionsModule', {}).get('interventions', [])]),
                'phase': ', '.join(design.get('phases', [])),
                'study_type': design.get('studyType'),
                'status': status_module.get('overallStatus'),
                'start_date': status_module.get('startDate'),
                'completion_date': status_module.get('completionDate'),
                'enrollment': design.get('enrollmentInfo', {}).get('count'),
                'sponsor': protocol.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name')
            })
        
        df = pd.DataFrame(df_data)
        
        # Show raw data
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Visualization section
        st.header("Data Visualizations")
        
        # Tab layout for different visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Status Distribution", "Phase Analysis", "Temporal Trends"])
        
        with viz_tab1:
            # Status distribution pie chart
            status_counts = df['status'].value_counts()
            fig1 = px.pie(status_counts, 
                         values=status_counts.values, 
                         names=status_counts.index,
                         title="Trial Status Distribution")
            st.plotly_chart(fig1)
            
        with viz_tab2:
            # Phase distribution bar chart
            if 'phase' in df.columns:
                phase_counts = df['phase'].value_counts()
                fig2 = px.bar(phase_counts, 
                              x=phase_counts.index, 
                              y=phase_counts.values,
                              title="Trial Phases Distribution",
                              labels={'x': 'Phase', 'y': 'Count'})
                st.plotly_chart(fig2)
            else:
                st.warning("Phase information not available in these results")
                
        with viz_tab3:
            # Temporal trends
            if 'start_date' in df.columns and pd.notnull(df['start_date']).any():
                df['year'] = pd.to_datetime(df['start_date']).dt.year
                yearly_counts = df['year'].value_counts().sort_index()
                fig3 = px.line(yearly_counts, 
                              x=yearly_counts.index, 
                              y=yearly_counts.values,
                              title="Trials Over Time",
                              labels={'x': 'Year', 'y': 'Number of Trials'})
                st.plotly_chart(fig3)
            else:
                st.warning("Start date information not available in these results")
        
        # Neural Network Analysis Section
        st.header("Neural Network Analysis")
        st.write("""
        This section provides basic neural network analysis to predict trial status 
        based on other features (phase, study type, etc.).
        """)
        
        # Prepare data for neural network
        nn_df = df.copy()
        
        # Select only columns with sufficient data
        nn_df = nn_df[['phase', 'study_type', 'status']].dropna()
        
        if len(nn_df) > 10:  # Need enough samples
            # Encode categorical variables
            le_phase = LabelEncoder()
            le_type = LabelEncoder()
            le_status = LabelEncoder()
            
            nn_df['phase_encoded'] = le_phase.fit_transform(nn_df['phase'])
            nn_df['type_encoded'] = le_type.fit_transform(nn_df['study_type'])
            nn_df['status_encoded'] = le_status.fit_transform(nn_df['status'])
            
            # Features and target
            X = nn_df[['phase_encoded', 'type_encoded']]
            y = nn_df['status_encoded']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train neural network
            mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
            mlp.fit(X_train, y_train)
            
            # Evaluate
            y_pred = mlp.predict(X_test)
            
            st.subheader("Model Performance")
            st.text(classification_report(y_test, y_pred, target_names=le_status.classes_))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig4 = px.imshow(cm,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=le_status.classes_,
                            y=le_status.classes_,
                            title="Confusion Matrix")
            st.plotly_chart(fig4)
            
            # Feature importance (simple approximation)
            st.subheader("Feature Importance")
            coefs = mlp.coefs_[0]
            importance = abs(coefs).mean(axis=1)
            
            fig5 = px.bar(x=['Phase', 'Study Type'], 
                         y=importance,
                         title="Relative Feature Importance",
                         labels={'x': 'Feature', 'y': 'Importance'})
            st.plotly_chart(fig5)
            
        else:
            st.warning("Not enough data with complete features for neural network analysis")
        
        # Download button
        st.download_button(
            label="Download Full Results as JSON",
            data=json.dumps(data, indent=2),
            file_name=f"clinical_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Download DataFrame as CSV
        st.download_button(
            label="Download Processed Data as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"clinical_trials_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")