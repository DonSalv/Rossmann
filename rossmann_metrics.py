import pandas as pd
import numpy as np


def _get_errors(actual: pd.Series, fitted: pd.Series) -> pd.Series:
    
    return actual - fitted


def _absolute_bias(errors: pd.Series, **kwargs) -> float:
    """
    Given a series of error (as difference with forecast minus actual value) it gives the absolute bias of the error.

    :param errors: The difference between forecast and actual values series
    :return: The absolute bias of the series
    """
    return errors.mean()


def _relative_bias(actual: pd.Series, errors: pd.Series) -> float:
    """
    Given a series of error (as difference with forecast minus actual value) it gives the relative bias of the error.

    :param actual: The series with the actual values of the series
    :param errors: The difference between forecast and actual values series
    :return: The relative bias of the series
    """
    return errors.sum() / actual.sum()


def _mape(actual: pd.Series, errors: pd.Series) -> float:
    """
    Given a series of error (as difference with forecast minus actual value) it gives the Mean Absolute Percentage
    Error (MAPE)  of the error.

    :param actual: The series with the actual values of the series
    :param errors: The difference between forecast and actual values series
    :return: The Mean Absolute Percentage Error (MAPE) of the series
    """
    return (errors / actual).sum() / len(actual)


def _absolute_mae(errors: pd.Series, **kwargs) -> float:
    """
    Given a series of error (as difference with forecast minus actual value) it gives the Absolute Mean Absolute Error
    (MAE) of the error.

    :param errors: The difference between forecast and actual values series
    :return: The Absolute Mean Absolute Error (MAE) of the series
    """
    return (errors.abs()).mean()


def _relative_mae(actual: pd.Series, errors: pd.Series) -> float:
    """
    Given a series of error (as difference with forecast minus actual value) it gives the Relative Mean Absolute Error
    (MAE) of the error.

    :param actual: The series with the actual values of the series
    :param errors: The difference between forecast and actual values series
    :return: The Relative Mean Absolute Error (MAE) of the series
    """
    return (errors.abs()).sum() / actual.sum()


def _absolute_rmse(errors: pd.Series, **kwargs) -> float:
    """
    Given a series of error (as difference with forecast minus actual value) it gives the Absolute Root Mean
    Squarer Error (RMSE) of the error.

    :param errors: The difference between forecast and actual values series
    :return: The Absolute Root Mean Squared Error (RMSE) of the series
    """
    return np.sqrt((errors ** 2).mean())


def _relative_rmse(actual: pd.Series, errors: pd.Series) -> float:
    """
    Given a series of error (as difference with forecast minus actual value) it gives the Relative Root Mean
    Squared Error (RMSE) of the error.

    :param actual: The series with the actual values of the series
    :param errors: The difference between forecast and actual values series
    :return: The Relative Root Mean Squared Error (RMSE) of the series
    """
    return np.sqrt((errors ** 2).mean()) / actual.mean()


def _mse(errors: pd.Series, **kwargs) -> float:
    """
    Given a series of error (as difference with forecast minus actual value) it gives the Mean Square Error
    of the error.

    :param errors: The difference between forecast and actual values series
    :return: The Mean Square Error  (MSE) of the series
    """

    return (errors ** 2).mean()


def get_forecast_evaluation(actual:pd.Series, fitted: pd.Series) -> dict:
    
    errors = _get_errors(actual = actual, fitted = fitted)
    
    return {"Bias" : _absolute_bias(errors),
            "R-Bias": _relative_bias(actual, errors),
            "MAPE": _mape(actual, errors),
            "MAE": _absolute_mae(errors),
            "R-MAE": _relative_mae(actual, errors),
            "RMSE" : _absolute_rmse(errors),
            "R-RMSE": _relative_rmse(actual, errors),
            "MSE": _mse(errors)
           }