import streamlit as st
import pandas as pd
from apputil import *

# Make the page wide for larger, aligned visuals
st.set_page_config(page_title="Titanic Visualizations", layout="wide")

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
left, mid, right = st.columns([0.06, 0.88, 0.06])

with mid:
    st.write(
    '''
    # Titanic Visualization 1

    **Question:** Did women and children in first class have significantly higher survival rates than adult men across all passenger classes, and how does age impact survival within each gender group?

    This visualization explores survival patterns by analyzing the intersection of passenger class, gender, and age groups (Child, Teen, Adult, Senior).
    '''
    )
    fig1 = visualize_demographic()
    st.plotly_chart(fig1, use_container_width=True)

    st.write(
    '''
    # Titanic Visualization 2

    **Question:** Is there a correlation between family size and ticket fare across different passenger classes, and do larger families tend to purchase cheaper tickets within their class?

    This visualization examines the relationship between family size (including siblings, spouses, parents, and children) and the average fare paid, broken down by passenger class.
    '''
    )
    fig2 = visualize_families()
    st.plotly_chart(fig2, use_container_width=True)

    st.write(
    '''
    # Titanic Visualization Bonus

    **Analysis:** How did traveling alone versus with family members affect survival chances across different passenger classes?

    This heatmap visualization reveals the complex relationship between family size categories and survival rates, showing that the optimal family size for survival varied significantly by passenger class.
    '''
    )
    fig3 = visualize_family_size()
    st.plotly_chart(fig3, use_container_width=True)