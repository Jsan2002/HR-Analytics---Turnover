import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="Employee Retention Dashboard", layout="wide")

# Title
st.title("Employee Retention Analytics Dashboard")

# Load data
@st.cache_data
def load_data():
    # Specify the encoding to handle potential issues
    df = pd.read_csv('turnover-data-set.csv', encoding='ISO-8859-1')
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 30, 35, 40, 100], 
                            labels=['<25', '25-30', '30-35', '35-40', '40+'])
    return df

df = load_data()

# Sidebar
st.sidebar.header('Dashboard Navigation')
page = st.sidebar.radio('Select Page', 
    ['Overview', 'Survival Analysis', 'Risk Analysis', 'Recommendations'])

if page == 'Overview':
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        st.metric("Average Tenure (months)", round(df['stag'].mean(), 1))
    with col3:
        st.metric("Turnover Rate", f"{(df['event'].mean()*100):.1f}%")
    with col4:
        st.metric("Avg Employee Age", round(df['age'].mean(), 1))

    # Age distribution
    st.subheader("Age Distribution by Department")
    fig_age = px.box(df, x='profession', y='age', color='profession',
                     title='Age Distribution across Departments')
    st.plotly_chart(fig_age, use_container_width=True)

    # Tenure by department
    st.subheader("Tenure Analysis by Department")
    fig_tenure = px.box(df, x='profession', y='stag', color='profession',
                       title='Tenure Distribution across Departments')
    st.plotly_chart(fig_tenure, use_container_width=True)

elif page == 'Survival Analysis':
    st.header("Survival Analysis")
    
    # Kaplan-Meier Survival Curves
    kmf = KaplanMeierFitter()
    
    # Select profession for analysis
    selected_prof = st.multiselect('Select Professions', 
                                 df['profession'].unique(),
                                 default=df['profession'].unique()[:3])
    
    fig = go.Figure()
    for prof in selected_prof:
        mask = (df['profession'] == prof)
        kmf.fit(df[mask]['stag'], 
                event_observed=df[mask]['event'],
                label=prof)
        fig.add_trace(go.Scatter(x=kmf.timeline,
                                y=kmf.survival_function_.values.flatten(),
                                name=prof))
    
    fig.update_layout(title='Survival Curves by Profession',
                     xaxis_title='Tenure (months)',
                     yaxis_title='Survival Probability')
    st.plotly_chart(fig, use_container_width=True)

elif page == 'Risk Analysis':
    st.header("Risk Analysis")

    # Calculate risk scores
    df['risk_score'] = (
        (df['age'] < 30).astype(int) * 2 +
        (df['anxiety'] > df['anxiety'].mean()).astype(int) * 1.5 +
        (df['stag'] < 12).astype(int) * 3
    )

    # Risk heatmap
    risk_matrix = pd.crosstab(df['profession'], df['industry'], 
                             values=df['risk_score'], aggfunc='mean')
    
    fig_heat = px.imshow(risk_matrix,
                        labels=dict(x="Industry", y="Profession", color="Risk Score"),
                        title="Risk Heatmap by Profession and Industry")
    st.plotly_chart(fig_heat, use_container_width=True)

    # High risk groups
    st.subheader("High Risk Groups")
    high_risk = df[df['risk_score'] >= df['risk_score'].quantile(0.9)]
    risk_prof = high_risk.groupby('profession').size().sort_values(ascending=True)
    
    fig_risk = px.bar(risk_prof, orientation='h',
                      title='Number of High-Risk Employees by Profession')
    st.plotly_chart(fig_risk, use_container_width=True)

elif page == 'Recommendations':
    st.header("Recommendations Based on Analysis")

    # Age Group Analysis
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 30, 35, 40, 100], labels=['<25', '25-30', '30-35', '35-40', '40+'])
    age_stats = df.groupby('age_group')['stag'].agg(['mean', 'count', 'std']).round(2)
    st.subheader("Age Group Analysis")
    st.write(age_stats)

    # Personality Traits Correlation with Tenure
    personality_traits = ['extraversion', 'independ', 'selfcontrol', 'anxiety', 'novator']
    personality_corr = df[personality_traits + ['stag']].corr()['stag'].sort_values(ascending=False)
    st.subheader("Personality Traits Correlation with Tenure")
    st.bar_chart(personality_corr)

    # Management Style Impact
    management_stats = df.groupby(['head_gender', 'coach'])['stag'].mean().round(2)
    st.subheader("Management Style Impact on Tenure")
    st.write(management_stats)

    # High Risk Groups
    df['retention_risk'] = (
        (df['age'] < 30).astype(int) * 2 +
        (df['anxiety'] > df['anxiety'].mean()).astype(int) * 1.5 +
        (df['stag'] < 12).astype(int) * 3
    )
    high_risk = df[df['retention_risk'] >= df['retention_risk'].quantile(0.9)]
    risk_profile = high_risk.groupby(['profession', 'industry']).size().sort_values(ascending=False).head()
    st.subheader("High Risk Groups (Top 10%)")
    st.write(risk_profile)

    # Recommendations
    st.subheader("Detailed Recommendations")
    recommendations = {
        "Age-Based Initiatives": [
            "Implement mentorship programs for employees under 30",
            "Create career development paths for mid-career professionals (30-40)",
            "Develop knowledge transfer programs leveraging experienced employees (40+)"
        ],
        "Personality-Based Strategies": [
            "Provide additional support for employees with high anxiety scores",
            "Create autonomous work opportunities for independent personalities",
            "Design team structures that balance different personality types"
        ],
        "Management Improvements": [
            "Expand coaching programs based on positive retention impact",
            "Implement regular feedback sessions",
            "Provide management training focused on retention strategies"
        ],
        "Industry-Specific Actions": [
            "Develop industry-specific retention programs",
            "Address unique challenges in high-turnover industries",
            "Create competitive compensation packages based on industry standards"
        ]
    }
    
    # Add observations based on the provided data
    st.subheader("Observations Based on Data")
    observations = [
        "Banks HR employees tend to stay longer.",
        "Manufacturing HR tend to leave early.",
        "Retail HR and IT can leave after 3 years."
    ]
    
    for observation in observations:
        st.write(f"- {observation}")

    for category, items in recommendations.items():
        st.write(f"**{category}:**")
        for item in items:
            st.write(f"- {item}")