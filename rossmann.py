import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pyarrow as pa

from plotly.subplots import make_subplots
from datetime import datetime

from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels.api as sm



def get_visitors(customers: int) -> int:
    
    conversion_rate = round(random.uniform(0.2, 0.4), 2)
    visitors = int(customers / conversion_rate)
    
    return visitors


def extract_data_from(data:pd.DataFrame) -> pd.DataFrame:

    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['WeekOfYear'] = data.index.weekofyear
    
    return data


def _set_operation_times(weekday: bool) -> tuple:
    
    if weekday == True:
        opens = random.choice(['08:00', '08:30', '09:00'])
        closes = random.choice(['19:00', '19:30', '20:00'])
    else:
        opens = random.choice(['09:00', '09:30', '10:00'])
        closes = random.choice(['14:00', '15:00', '16:00'])
    
    return opens, closes


def get_scheduled_data(data: pd.DataFrame) -> pd.DataFrame:
    
    stores = data.Store.unique().tolist()

    columns = ['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                'Visitors', 'Year', 'Month', 'Day', 'WeekOfYear', 'SalePerVisitor', 'ScheduleTime']

    scheduled_data = pd.DataFrame(columns=columns)

    for store in stores:
        np.random.seed(store)
        opens_at, closes_at = _set_operation_times(weekday=True)
        opens_wk, closes_wk = _set_operation_times(weekday=False)

        tempo_df = data[data['Store'] == store]
        tempo_df['ScheduleTime'] = tempo_df['DayOfWeek'].apply(lambda x: (opens_wk, closes_wk) if (x == 6) else (np.nan if (x == 7) else (opens_at, closes_at)))
        
        scheduled_data =pd.concat([scheduled_data, tempo_df], ignore_index=False) #, sort=False)

    return scheduled_data


def get_time_difference(time1: str, time2: str) -> int:

    if time1 is np.nan or time2 is np.nan:
        return np.nan
    else:
        frmat = '%H:%M'
        time_delta = datetime.strptime(time2, frmat) - datetime.strptime(time1, frmat)
    
    return time_delta.seconds//60


def write_parquet_with_schema(data:pd.DataFrame, path: str):
    
    df = data.copy()
    
    df = df.reset_index()
    df.rename(columns={'index': 'Date'}, inplace=True)

    schema = pa.schema([pa.field('Date', pa.date32()),
                        pa.field('Store', pa.int16()), 
                        pa.field('DayOfWeek', pa.int8()), 
                        pa.field('Sales', pa.float64()), 
                        pa.field('Customers', pa.int32()),
                        pa.field('Open', pa.int8()), 
                        pa.field('Promo', pa.int8()),
                        pa.field('StateHoliday', pa.string()), 
                        pa.field('SchoolHoliday', pa.int8()),
                        pa.field('Visitors', pa.int32()), 
                        pa.field('Year', pa.int16()), 
                        pa.field('Month', pa.int16()), 
                        pa.field('Day', pa.int16()),
                        pa.field('WeekOfYear', pa.int16()), 
                        pa.field('ScheduleTime', pa.list_(pa.string())),
                        pa.field('EffectiveTime', pa.int32())
                       ])
    
    df.to_parquet(path, schema=schema)


### Plots
def get_cumulative_distribution_graph(data:pd.DataFrame, variable:str) -> go.Figure:
    cdf = ECDF(data[variable])

    fig = go.Figure(go.Scatter(x=cdf.x, y=cdf.y))
    # fig.add_vline(x=10000, line_width=2, line_dash="dash", line_color="green")
    fig.add_hline(y=0.80, line_width=2, line_dash="dash", line_color="green")
    fig.update_layout(title=f"Cumulative distribution: {variable}",
                     xaxis_title=f"{variable}",
                     yaxis_title='Percentage')

    return fig

def get_detailed_visitors_behaviour_per_store(data:pd.DataFrame, store_type: str) -> go.Figure:
    
    df = data[['Month', 'Visitors', 'StoreType', 'Promo']].groupby(['StoreType', 'Promo', 'Month']).mean()

    # Visualize per StoreType with and without promotion
    nopromo = df.loc[(df.index.get_level_values('StoreType') == store_type) & (df.index.get_level_values('Promo') == 0)]
    promo = df.loc[(df.index.get_level_values('StoreType') == store_type) & (df.index.get_level_values('Promo') == 1)]

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
              'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure([go.Scatter(x=months, y=nopromo.reset_index(drop=True).Visitors, 
                                name='No Promo', hovertemplate="Month: %{x} <br>"+"Customers: %{y}"),
                     go.Scatter(x=months, y=promo.reset_index(drop=True).Visitors, 
                                name='Promo', hovertemplate="Month: %{x} <br>"+"Customers: %{y}")])
    fig.update_layout(title =f"Average number of visitors per month under promotion influence - Store Type {store_type}",
                     xaxis_title = 'Month',
                     yaxis_title = 'Visitors')

    return fig


def get_detailed_visitors_behaviour_per_promo(data:pd.DataFrame) -> go.Figure:
    
    df = data[['DayOfWeek', 'Visitors', 'Promo', 'Promo2']].groupby(['Promo', 'Promo2', 'DayOfWeek']).mean()

    # Visualize per StoreType with and without promotion
    col1row1 = df.loc[(df.index.get_level_values('Promo') == 0) & (df.index.get_level_values('Promo2') == 0)]
    col1row2 = df.loc[(df.index.get_level_values('Promo') == 0) & (df.index.get_level_values('Promo2') == 1)]
    
    col2row1 = df.loc[(df.index.get_level_values('Promo') == 1) & (df.index.get_level_values('Promo2') == 0)]
    col2row2 = df.loc[(df.index.get_level_values('Promo') == 1) & (df.index.get_level_values('Promo2') == 1)]
    
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("No promo (Discontinued)", "Promotion (Discontinued)", "No promo (Continued)", "Promotion (Continued)")
                       )

    days = ['Mon', 'Tue', 'Wed', 'Thr', 'Fri', 'Sat']
    
    fig.add_trace(go.Scatter(x=days, y=col1row1.reset_index(drop=True).Visitors, 
                             name='No promo and Discontinued Promo', 
                             hovertemplate="Day: %{x} <br>"+"Visitors: %{y}"), 
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=days, y=col1row2.reset_index(drop=True).Visitors, 
                             name='No promo and Continued Promo', 
                             hovertemplate="Day: %{x} <br>"+"Visitors: %{y}"), 
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=days, y=col2row1.reset_index(drop=True).Visitors, 
                             name='Promo and Discontinued Promo', 
                             hovertemplate="Day: %{x} <br>"+"Visitors: %{y}"), 
                  row=1, col=2)
    
    fig.add_trace(go.Scatter(x=days, y=col2row2.reset_index(drop=True).Visitors, 
                             name='Promo and Continued Promo', 
                             hovertemplate="Day: %{x} <br>"+"Visitors: %{y}"), 
                  row=2, col=2)
    
    # Update xaxis properties
    fig.update_xaxes(title_text="Day of the Week", row=2, col=1)
    fig.update_xaxes(title_text="Day of the Week", row=2, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="Visitors", row=1, col=1)
    fig.update_yaxes(title_text="Visitors", row=2, col=1)
    
    
    fig.update_layout(title = "Average number of visitors per day under promotion influence",
                      height = 600,
                      showlegend=False)

    return fig


def get_sales_performance_by_visitors_ratio(data: pd.DataFrame, worst_store: int, best_store: int) -> go.Figure:

    fig = go.Figure([go.Scatter(x=data[(data.Store == best_store)&(data.Year == 2015)].Sales.index, 
                                y=data[(data.Store == best_store)&(data.Year == 2015)].Sales.values, 
                                name=f"Store {best_store}", 
                                hovertemplate="Total Sales: $%{y:.2f}"),
                     go.Scatter(x=data[(data.Store == worst_store)&(data.Year == 2015)].Sales.index, 
                                y=data[(data.Store == worst_store)&(data.Year == 2015)].Sales.values, 
                                name=f"Store {worst_store}",
                                hovertemplate="Total Sales: $%{y:.2f}"),
                     ])
    

    fig.update_layout(title = f"Sales Performance: Store {worst_store} vs Store {best_store}",
                      height = 450,
                      hovermode="x unified",
                      xaxis_title= "Year 2015",
                      yaxis_title="Sales")

    return fig


def get_time_series_decomposition(time_series: pd.DataFrame, model: str = 'additive'):
    
    time_series = time_series.fillna(0)
    
    decomposition = sm.tsa.seasonal_decompose(time_series, 
                                              model=model, freq=52)
    
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Observed", "Trend", "Resid", "Seasonal"))
    
    # Observed or Actual values
    fig.add_trace(go.Scatter(x=decomposition.observed.index, 
                             y=decomposition.observed.values, 
                             marker_color='cornflowerblue'), row=1, col=1)
    # Trend decomposition
    fig.add_trace(go.Scatter(x=decomposition.trend.index, 
                             y=decomposition.trend.values,
                             marker_color='cornflowerblue'), row=1, col=2)
    
    # Residuals
    fig.add_trace(go.Scatter(x=decomposition.resid.index, 
                             y=decomposition.resid.values, 
                             marker_color='cornflowerblue'), row=2, col=1)

    # Seasonal decomposition
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, 
                             y=decomposition.seasonal.values, 
                             marker_color='cornflowerblue'), row=2, col=2)

    fig.update_layout(height=500, 
                      autosize=True,
                      title_text="Time Series Decomposition",
                      showlegend=False)

    return fig