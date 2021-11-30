import plotly.graph_objects as go
import pandas as pd
import datetime as dt


def get_plot_prediction_from(forecast: pd.DataFrame, 
                             actual: pd.DataFrame):
    
    fig = go.Figure([
        go.Scatter(
            name='Forecast',
            x=forecast.ds,
            y=forecast.yhat,
            mode='lines',
            hovertemplate="Sales: $%{y:.2f}",
            line=dict(color='rgb(178, 34, 34 )'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=forecast.ds,
            y=forecast.yhat_upper,
            mode='lines',
            marker=dict(color="#444"),
            hoverinfo="skip",
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=forecast.ds,
            y=forecast.yhat_lower,
            marker=dict(color="#444"),
            line=dict(width=0),
            hoverinfo="skip",
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        ),
        go.Scatter(name='Actual Demand',
                   x=actual.ds, 
                   y=actual.y,
                   mode='lines',
                   hovertemplate="Sales: $%{y:.2f}",
                   line=dict(color='rgb(100, 149, 237)'),
                  )
    ])
    
    fig.update_layout(
        height=500,
        yaxis_title='Demand (confidence band)',
        title="Actual Demand vs Prophet's prediction",
        hovermode="x"
    )
    
    return fig


def get_plot_with_forecast_from(forecast: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> go.Figure:
    
    fig = go.Figure([
        go.Scatter(
            name='Forecast',
            x=forecast.ds,
            y=forecast.yhat,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=forecast.ds,
            y=forecast.yhat_upper,
            mode='lines',
            marker=dict(color="#444"),
            hoverinfo="skip",
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=forecast.ds,
            y=forecast.yhat_lower,
            marker=dict(color="#444"),
            line=dict(width=0),
            hoverinfo="skip",
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        ),
        go.Scatter(name='Test Demand',
                   x=test.ds, 
                   y=test.y,
                   mode='lines',
                   hovertemplate='%{y:.3f}'
                  ),
        go.Scatter(name='Train Demand',
                   x=train.ds, 
                   y=train.y,
                   mode='lines',
                   hovertemplate='%{y:.3f}')
    ])
    fig.update_layout(
        height=500,
        yaxis_title='Demand (confidence band)',
        title="Actual Demand vs Prophet's prediction",
        hovermode="x"
    )

    return fig

## Seasonalities

def get_yearly_seasonality_from(forecast: pd.DataFrame) -> pd.DataFrame:
    yearly = forecast[['ds', 'yearly']]
    yearly['month'] = yearly['ds'].apply(lambda x: x.month_name())

    summary = yearly.groupby('month').mean()
    summary.reset_index(inplace=True)

    sorting = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
           'August', 'September', 'October', 'November', 'December']

    summary.index = pd.CategoricalIndex(summary['month'], 
                                        categories=sorting, 
                                        ordered=True)

    summary = summary.sort_index().reset_index(drop=True)
    
    return summary


def get_weekly_seasonality_from(forecast: pd.DataFrame) -> pd.DataFrame:
    
    weekly = forecast[['ds', 'weekly']]
    weekly['day'] = weekly.ds.apply(lambda x: x.weekday_name)
    summary = weekly.groupby('day').mean()
    summary.reset_index(inplace=True)

    sorting = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    summary.index = pd.CategoricalIndex(summary['day'], 
                                        categories=sorting, 
                                        ordered=True)

    summary = summary.sort_index().reset_index(drop=True)
    
    return summary


def get_yearly_seasonality_plot_from(forecast: pd.DataFrame) -> go.Figure:
    
    summary = get_yearly_seasonality_from(forecast=forecast)
    
    fig = go.Figure()

    fig.add_trace(go.Line(x=summary.month,
                         y=summary.yearly)
                 )

    fig.update_layout(title='Seasonality for Demand: Montlhy Level',
                      xaxis_title='Month',
                      yaxis_title='Yearly Trend average',
                      height=500,
                     )
    return fig


def get_weekly_seasonality_plot_from(forecast: pd.DataFrame) -> go.Figure:
    
    summary = get_weekly_seasonality_from(forecast=forecast)
    
    fig = go.Figure()

    fig.add_trace(go.Line(x=summary.day,
                         y=summary.weekly)
                 )

    fig.update_layout(title='Seasonality for Demand: Day Level',
                      xaxis_title='Weekday',
                      yaxis_title='Weekly Trend average',
                      height=500,
                     )
    
    return fig

def get_plot_for_missing(forecast: pd.DataFrame, name: str) -> go.Figure :
    
    fig = go.Figure()
    fig.add_traces([go.Scatter(name='Acutal',
                             x=forecast.index, 
                             y=forecast.actual,
                             mode='lines',
                             hovertemplate='Demand: %{y:.3f}' '<br>Date: %{x}'),
                    go.Scatter(name=name,
                         x=forecast.index, 
                         y=forecast[name],
                         mode='lines',
                         hovertemplate='Demand: %{y:.3f}' '<br>Date: %{x}')]
             )

    fig.update_layout(height=500,
                  title='Demand from 2016-01-09 to 2018-10-18', 
                  xaxis={'title':'Date'},
                  yaxis_title='Actual',
                 )

    return fig