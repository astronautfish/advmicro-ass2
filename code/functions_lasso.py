import pandas as pd 
import numpy as np 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
# import matplotlib.patheffects as pe


def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_tilde = (X - mu)/sigma
    
    return X_tilde

def BRT(X_tilde,y):
    (N,p) = X_tilde.shape
    sigma = np.std(y, ddof=1)
    c=1.1
    alpha=0.05

    penalty_BRT= (sigma*c)/np.sqrt(N)*norm.ppf(1-alpha/(2*p)) 

    return penalty_BRT

def BCCH(X_tilde,y):
    (N,p) = X_tilde.shape
    c=1.05
    alpha=0.05

    yXscale = (np.max((X_tilde.T ** 2) @ ((y-np.mean(y)) ** 2) / N)) ** 0.5
    penalty_pilot = c / np.sqrt(N) * norm.ppf(1-alpha/(2*p)) * yXscale
    pred = Lasso(alpha=penalty_pilot).fit(X_tilde,y).predict(X_tilde)
    eps = y - pred 
    epsXscale = (np.max((X_tilde.T ** 2) @ (eps ** 2) / N)) ** 0.5
    penalty_BCCH = c*norm.ppf(1-alpha/(2*p))*epsXscale/np.sqrt(N)

    return penalty_BCCH

def penalty_grid_gen(start=0.01, stop=80000, num=50, **kwargs):
    penalty_grid = np.geomspace(start, stop, num=num, **kwargs)

    return penalty_grid

def penalty_estimate(X, y, penalty_type: str, alpha=0.05, c=1.1, CV=5):
    n,p = X.shape

    if penalty_type == "BRT":
        sigma = np.std(y)
        max_term = np.max((1/n) * np.sum((X**2),axis=0))**0.5 # Note: this equals 1 for standardized data, and is therefore not necessary to compute or include in the equation
        penalty = c * sigma / np.sqrt(n) * norm.ppf(1 - alpha / (2*p)) * max_term # Note: Have divided by 2 due to Python definition of Lasso
    
    elif penalty_type == "BCCH":
        yXscale = (np.max((X.T ** 2) @ ((y-np.mean(y)) ** 2) / n)) ** 0.5
        penalty_pilot = c / np.sqrt(n) * norm.ppf(1-alpha/(2*p)) * yXscale # Note: Have divided by 2 due to Python definition of Lasso
        pred = Lasso(alpha=penalty_pilot).fit(X,y).predict(X)
        eps = y - pred 
        epsXscale = (np.max((X.T ** 2) @ (eps ** 2) / n)) ** 0.5
        penalty = c*norm.ppf(1-alpha/(2*p))*epsXscale/np.sqrt(n)
    else:
        raise TypeError("BRT or BCCH is not defined")
    
    return penalty

def get_selected_var(selected_variables, xs):
    xs_selected = []
    for i in range(0,len(selected_variables)):
        if selected_variables[i] == True:
            xs_selected.append(i)
    xs_varname = []

    for var_index in xs_selected:
        xs_varname.append(xs[var_index])

    return xs_varname

# def coefs_Lasso(X, y, penalty_type: str, penalty=None, CV=5):
#     """
#     penalty should be penalty_grid if a grid is used. For CV, a five fold is used. BRT and BCCH, use normal penalty from penalty_estimate().
#     """
#     if penalty_type == "Grid":
#         coefs = []
#         for lamb in penalty:
#             fit = Lasso(alpha = lamb).fit(X,y) 
#             coefs.append(fit.coef_)
    
#     elif penalty_type == "CV":
#         fit_CV = LassoCV(cv=CV, alphas=penalty).fit(X,y)
#         penalty_CV = fit_CV.alpha_ 
#         coefs = fit_CV.coef_
    
#     elif penalty_type == "BCCH" or penalty_type == "BRT":
#         fit_BCCH = Lasso(alpha=penalty).fit(X,y)
#         coefs = fit_BCCH.coef_

#     selected_variables = (coefs!=0)
#     return coefs, selected_variables


def plot_lasso_path(penalty_grid, coefs, legends, title="Lasso Path", vlines: dict = None):
    """
    Plots the coefficients as a function of the penalty parameter for Lasso regression.

    Parameters:
    penalty_grid (array-like): The penalty parameter values.
    coefs (array-like): The estimated coefficients for each penalty value.
    legends (list): The labels for each coefficient estimate.
    vlines (dict, optional): A dictionary of vertical lines to add to the plot. The keys are the names of the lines and the values are the penalty values where the lines should be drawn.
    
    """
    # Initiate figure 
    fig, ax = plt.subplots()

    # Plot coefficients as a function of the penalty parameter
    ax.plot(penalty_grid, coefs)

    # Set log scale for the x-axis
    ax.set_xscale('log')
    ax.set_xlim([np.min(penalty_grid), 2])

    # Add labels
    plt.xlabel(r'Penalty, $\lambda$')
    plt.ylabel(r'Estimates, $\widehat{\beta}_j(\lambda)$')
    plt.title(title)

    # Add legends
    lgd=ax.legend(legends,loc=(1.04,-0.3))
    
    
    # Add vertical lines
    if vlines is not None:
        for name, penalty in vlines.items():
            ax.axvline(x=penalty, linestyle='--', color='grey')
            plt.text(penalty,min(coefs[0]),name,rotation=90)

    # Display plot
    plt.show()
    plt.close()
