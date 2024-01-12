import numpy as np
from sklearn.metrics import r2_score

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).
    """
    return np.mean(np.abs(y_true - y_pred))

def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE).
    """
    return np.mean((y_true - y_pred)**2)

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).
    """
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_r2_score(y_true, y_pred):
    """
    Calculate R-squared (R2) score.
    """
    return r2_score(y_true, y_pred)


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100