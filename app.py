import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from apputil import *

# Make the page wide for larger, aligned visuals
st.set_page_config(page_title="Titanic Visualizations", layout="wide")

# Layout: narrow gutters with a wide center column for alignment
left, mid, right = st.columns([0.06, 0.88, 0.06])

with mid:
    st.write(
        '''
        # Titanic Visualization 1

        **Question:** Did women and children in first class have significantly higher survival rates than adult men across all passenger classes, and how does age impact survival within each gender group?

        This visualization explores survival patterns by analyzing the intersection of passenger class, gender, and age groups (Child, Teen, Adult, Senior).
        '''
    )
    # From apputil.py (compatible)
    fig1 = visualize_demographic()
    st.plotly_chart(fig1, use_container_width=True)

    st.write(
        '''
        # Titanic Visualization 2

        **Question:** Is there a correlation between family size and ticket fare across different passenger classes, and do larger families tend to purchase cheaper tickets within their class?

        This visualization examines the relationship between family size (including siblings, spouses, parents, and children) and the average fare paid, broken down by passenger class.
        '''
    )
    # Build the families scatter here (avoid calling visualize_families() which
    # calls family_groups with an argument in apputil.py)
    fam_df = family_groups()  # uses apputil.py; no arguments

    fig2 = px.scatter(
        fam_df,
        x='family_size',
        y='avg_fare',
        size='n_passengers',
        color='Pclass',
        title='Average Fare by Family Size and Passenger Class<br><sub>Do larger families pay less per person?</sub>',
        labels={
            'family_size': 'Family Size (including self)',
            'avg_fare': 'Average Fare (£)',
            'n_passengers': 'Number of Passengers',
            'Pclass': 'Passenger Class'
        },
        color_continuous_scale='Viridis',
        hover_data=['min_fare', 'max_fare'],
        size_max=50,
        height=700
    )

    # Add per-class linear trendlines
    for pclass in sorted(fam_df['Pclass'].dropna().unique()):
        class_data = fam_df[fam_df['Pclass'] == pclass]
        if len(class_data) > 1:
            z = np.polyfit(class_data['family_size'], class_data['avg_fare'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(class_data['family_size'].min(), class_data['family_size'].max(), 100)
            fig2.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name=f'Class {pclass} trend',
                    line=dict(dash='dash', width=2),
                    showlegend=True
                )
            )

    fig2.update_layout(
        xaxis=dict(dtick=1, title='Family Size (including self)'),
        yaxis=dict(title='Average Fare (£)', gridcolor='lightgray'),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=40, r=30, t=100, b=40),
        font=dict(size=14)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.write(
        '''
        # Titanic Visualization Bonus

        **Analysis:** How did traveling alone versus with family members affect survival chances across different passenger classes?

        This heatmap visualization reveals the complex relationship between family size categories and survival rates, showing that the optimal family size for survival varied significantly by passenger class.
        '''
    )
    # From apputil.py (compatible)
    fig3 = visualize_family_size()
    st.plotly_chart(fig3, use_container_width=True)
