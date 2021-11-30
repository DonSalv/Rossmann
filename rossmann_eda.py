import pandas as pd
import numpy as np
import phik
import itertools

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go


def get_time_frame_info(data: pd.DataFrame, date_column: str):
    """ 
    Given a dataframe and the name of a column with datetime information it gives the starting and finishing dates with 
    the total of days in the time frame.
    
    :param data: A pandas dataframe with date information
    :param date_column: The name of a column with time information. Expected in format YYYY-mm-dd
    return: Print a summary of starting and finishing date with the days elapsed in the period
    
    """
    
    df = data[date_column]
    
    min_date = df.min()
    max_date = df.max()
    range_date = max_date - min_date

    print(f"The time frame starts on {min_date.strftime('%Y-%m-%d')} and conclude on {max_date.strftime('%Y-%m-%d')}. A total of {range_date.days} days")
    
    
## Plots ##
def bar_plot(data: pd.DataFrame) -> go.Figure:
    fig = px.bar(data.count().rename('count'))
    fig.update_traces(marker_color='#28587B',
                      hovertemplate = 'Feature: %{x}' + '<br> Non-nulls: %{y:5f}',
                      name='',
                     showlegend=False,)
    fig.update_layout(yaxis_title = 'Counts',
                      xaxis_title='',
                      title='Missing Values - Bar',
                      plot_bgcolor='rgba(0,0,0,0)')
    
    return fig


def _pearson_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    correlation_matrix = 1 - data.corr()
    rows_with_nan_vals = [index for index, row in correlation_matrix.iterrows() if row.isnull().all()]
    correlation_matrix = correlation_matrix.dropna(how='all')
    correlation_matrix.drop(columns=rows_with_nan_vals, inplace=True)
    
    return correlation_matrix

def correlation_plot(data: pd.DataFrame, correlation: str = 'pearson') -> go.Figure :
    """ 
    Given a correlation object, it gives a heatmap plot for the correlation

    :param correlation: A Correlation instance for the coefficients {spearman, pearson, kendall, cramers, phi_k}
    :return: A graphic object Figure of type heatmap for the correlation
    
    """
    
    if correlation == 'pearson':
        name = 'Pearson Correlation'
        matrix = _pearson_correlation_matrix(data)
    else:
        name = 'Phi(k) Correlation'
        matrix = data.phik_matrix()

    coefficients = matrix.values
    variables = matrix.columns.tolist()
    
    fig = go.Figure(go.Heatmap(z=coefficients,
                               x=variables,
                               y=variables,
                               # type='heatmap',
                               colorscale='Blues'))
    
    fig.update_layout(yaxis=dict(autorange = 'reversed'),
                      title=name,
                     height=400)
    
    return fig


def plot_monthly_kpis_under_promo(data: pd.DataFrame, kpi:str ) -> go.Figure:
    
    hover_label = kpi.capitalize()
    
    df = data.groupby(['StoreType', 'Promo', 'Month']).mean()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    storetypes = ['a', 'b', 'c', 'd']
    promotions = [0, 1]

    fig = make_subplots(rows=2, cols=4, 
                        subplot_titles=("No Promo | Store a", "No Promo | Store b", "No Promo | Store c", "No Promo | Store d",
                                        "Promo | Store a", "Promo | Store b", "Promo | Store c", "Promo | Store d"))
    i=0
    for element in itertools.product(promotions, storetypes):
        plot_df = df.loc[(df.index.get_level_values('StoreType') == element[1]) & (df.index.get_level_values('Promo') == element[0])]
        fig.add_trace(go.Scatter(x=months, y=plot_df.reset_index(drop=True)[kpi],
                                 name=f'Store Type: {element[1]}',
                                 marker_color='cornflowerblue',
                                 hovertemplate="Month: %{x} <br>"+ f"{hover_label}: " + "%{y}"),
                      row = element[0]+1, col = i%4+1,)
        i +=1

    fig.update_layout(title =f"Average {hover_label} per month under promotion influence",
                      height= 600,
                      autosize=True,
                      showlegend = False)

    return fig
    
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.figure_factory as ff
# from scipy.cluster.hierarchy import linkage
# from pydantic import BaseModel
# from typing import Dict, Optional