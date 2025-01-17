import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(layout="wide")
with open('logreg_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('transformer.pkl', 'rb') as file:
    transformer = pickle.load(file)
categorical_features = [
    'gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice', 'multiplelines', 
    'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 
    'techsupport', 'streamingtv', 'streamingmovies', 'contract', 
    'paperlessbilling', 'paymentmethod', 'churn'
]
numerical_features = ['tenure', 'monthlycharges', 'totalcharges']
category_options = {
    'gender': ['Female', 'Male'],
    'seniorcitizen': [0, 1],
    'partner': ['Yes', 'No'],
    'dependents': ['Yes', 'No'],
    'phoneservice': ['Yes', 'No'],
    'multiplelines': ['No phone service', 'No', 'Yes'],
    'internetservice': ['DSL', 'Fiber optic', 'No'],
    'onlinesecurity': ['Yes', 'No', 'No internet service'],
    'onlinebackup': ['Yes', 'No', 'No internet service'],
    'deviceprotection': ['Yes', 'No', 'No internet service'],
    'techsupport': ['Yes', 'No', 'No internet service'],
    'streamingtv': ['Yes', 'No', 'No internet service'],
    'streamingmovies': ['Yes', 'No', 'No internet service'],
    'contract': ['Month-to-month', 'One year', 'Two year'],
    'paperlessbilling': ['Yes', 'No'],
    'paymentmethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
}

default_values = {
    'gender': 'Female',
    'seniorcitizen': 0,
    'partner': 'Yes',
    'dependents': 'No',
    'phoneservice': 'Yes',
    'multiplelines': 'No phone service',
    'internetservice': 'DSL',
    'onlinesecurity': 'No',
    'onlinebackup': 'Yes',
    'deviceprotection': 'Yes',
    'techsupport': 'No',
    'streamingtv': 'No',
    'streamingmovies': 'No',
    'contract': 'Month-to-month',
    'paperlessbilling': 'Yes',
    'paymentmethod': 'Electronic check',
    'tenure': 10,
    'monthlycharges': 60.0,
    'totalcharges': 600.0
}

def create_gauge_chart(churn_probability):
    """Create a speedometer-like gauge chart using plotly"""
    
    colors = [
        '#00B050',  
        '#FFA500', 
        '#FF0000'  
    ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_probability * 100,
        domain={'x': [0, 0.3], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkgray"},
            'steps': [
                {'range': [0, 30], 'color': colors[0]},
                {'range': [30, 65], 'color': colors[1]},
                {'range': [65, 100], 'color': colors[2]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': churn_probability * 100
            }
        },
        title={'text': "Churn Risk", 'font': {'size': 15}}
    ))
    
    fig.update_layout(
        height=180,
        width=180,
        margin=dict(l=420, r=0, t=50, b=5), 
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font={'color': "white", 'family': "Arial"}
    )
    
    return fig

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        .element-container {
            margin-bottom: 0px;
        }
        .stButton>button {
            margin-bottom: 0rem;
        }
        div[data-testid="stVerticalBlock"] > div:first-of-type {
            margin-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("Customer Churn Prediction")

# Sidebar setup
st.sidebar.header("User Inputs")

# Collect user inputs
user_inputs = {}
for feature in categorical_features:
    if feature == 'seniorcitizen':
        user_inputs[feature] = st.sidebar.selectbox(
            feature,
            options=category_options[feature],
            index=category_options[feature].index(default_values[feature]),
            format_func=lambda x: 'Yes' if x == 1 else 'No'
        )
    elif feature in category_options:
        user_inputs[feature] = st.sidebar.selectbox(
            feature,
            options=category_options[feature],
            index=category_options[feature].index(default_values[feature])
        )

for feature in numerical_features:
    user_inputs[feature] = st.sidebar.number_input(
        feature,
        value=default_values[feature],
        step=1 if feature == 'tenure' else 0.01
    )

# Convert inputs to a DataFrame
input_data = pd.DataFrame([user_inputs])
input_data['seniorcitizen'] = input_data['seniorcitizen'].astype('object')

# Apply preprocessing using the transformer
input_data_transformed = transformer.transform(input_data)

# Prediction Section
st.header("Prediction")
if st.button("Predict"):
    probabilities = model.predict_proba(input_data_transformed)[0]
    churn_prob = probabilities[1]
    
    gauge_chart = create_gauge_chart(churn_prob)
    st.plotly_chart(gauge_chart, use_container_width=True)
    
    # Display prediction information
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.write(f"Probability of Churn: {probabilities[1]:.2%}")
    st.write(f"Probability of Retention: {probabilities[0]:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display final prediction based on probability ranges
    if churn_prob * 100 < 30:
        st.success(f"### Final Prediction: **Low Churn Risk**")
    elif churn_prob * 100 < 65:
        st.warning(f"### Final Prediction: **Moderate Churn Risk**")
    else:
        st.error(f"### Final Prediction: **High Churn Risk**")