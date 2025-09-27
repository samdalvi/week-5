import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def survival_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    age_bins = [0, 12, 19, 59, float('inf')]
    age_labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
    grouped = df.groupby(['Pclass', 'Sex', 'age_group'], observed=True)
    result = grouped.agg(
        n_passengers=('PassengerId', 'count'),
        n_survivors=('Survived', 'sum')
    ).reset_index()
    result['survival_rate'] = result['n_survivors'] / result['n_passengers']
    result = result.sort_values(['Pclass', 'Sex', 'age_group'])
    return result


def family_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    grouped = df.groupby(['family_size', 'Pclass'])
    result = grouped.agg(
        n_passengers=('PassengerId', 'count'),
        avg_fare=('Fare', 'mean'),
        min_fare=('Fare', 'min'),
        max_fare=('Fare', 'max')
    ).reset_index()
    result = result.sort_values(['Pclass', 'family_size'])
    return result


def last_names(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df['last_name'] = df['Name'].str.split(',').str[0].str.strip()
    return df['last_name'].value_counts()


def determine_age_division(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    median_ages = df.groupby('Pclass')['Age'].transform('median')
    df['older_passenger'] = df['Age'] > median_ages
    return df


def visualize_demographic():
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    data = survival_demographics(df)

    fig = px.bar(
        data,
        x='age_group',
        y='survival_rate',
        color='Sex',
        facet_col='Pclass',
        title='Survival Rates by Class, Sex, and Age Group<br><sub>Did women and children in first class survive at higher rates?</sub>',
        labels={
            'survival_rate': 'Survival Rate',
            'age_group': 'Age Group',
            'Pclass': 'Passenger Class',
            'Sex': 'Gender'
        },
        color_discrete_map={'male': '#636EFA', 'female': '#EF553B'},
        barmode='group',
        height=700  # bigger
    )

    fig.update_layout(
        yaxis_tickformat='.0%',
        showlegend=True,
        legend=dict(title="Gender", orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=100, b=40),
        font=dict(size=14)
    )
    fig.update_yaxes(tickformat='.0%', range=[0, 1])
    fig.for_each_annotation(lambda a: a.update(text=f"Class {a.text.split('=')[1]}"))
    return fig


def visualize_families():
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    data = family_groups(df)

    fig = px.scatter(
        data,
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
        height=700  # bigger
    )

    for pclass in sorted(data['Pclass'].unique()):
        class_data = data[data['Pclass'] == pclass]
        if len(class_data) > 1:
            z = np.polyfit(class_data['family_size'], class_data['avg_fare'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(class_data['family_size'].min(), class_data['family_size'].max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name=f'Class {pclass} trend',
                    line=dict(dash='dash', width=2),
                    showlegend=True
                )
            )

    fig.update_layout(
        xaxis=dict(dtick=1, title='Family Size (including self)'),
        yaxis=dict(title='Average Fare (£)', gridcolor='lightgray'),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=40, r=30, t=100, b=40),
        font=dict(size=14)
    )
    return fig


def visualize_family_size():
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['family_category'] = pd.cut(
        df['family_size'],
        bins=[0, 1, 2, 4, 11],
        labels=['Solo (1)', 'Small (2)', 'Medium (3-4)', 'Large (5+)'],
        include_lowest=True
    )

    analysis = df.groupby(['family_category', 'Pclass']).agg(
        n_passengers=('PassengerId', 'count'),
        survival_rate=('Survived', 'mean'),
        avg_age=('Age', 'mean')
    ).reset_index()

    pivot_data = analysis.pivot(index='family_category', columns='Pclass', values='survival_rate')

    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=['Class 1', 'Class 2', 'Class 3'],
        y=pivot_data.index,
        colorscale='RdYlGn',
        text=[[f'{val:.1%}' for val in row] for row in pivot_data.values],
        texttemplate='%{text}',
        textfont={"size": 16},
        colorbar=dict(title='Survival Rate', tickformat='.0%'),
        hovertemplate='Family Size: %{y}<br>%{x}<br>Survival Rate: %{z:.1%}<extra></extra>'
    ))

    fig.update_layout(
        title='Survival Rates by Family Size and Passenger Class<br><sub>How did traveling alone vs. with family affect survival chances?</sub>',
        xaxis_title='Passenger Class',
        yaxis_title='Family Size Category',
        height=700,  # bigger
        xaxis={'side': 'top'},
        font=dict(size=14),
        margin=dict(l=40, r=30, t=100, b=40)
    )

    fig.add_annotation(
        text="Solo travelers in 1st class had high survival rates",
        xref="x", yref="y",
        x=0, y=0,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="black",
        ax=-50, ay=30,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    return fig
