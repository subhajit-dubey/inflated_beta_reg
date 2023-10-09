"""
Inflated Beta Model
"""

import numpy as np
import scipy.special as sp
from scipy.optimize import minimize
from scipy.stats import t
import numdifftools as ndt
import pandas as pd

class InflatedBeta:
    r"""Class implementing Inflated Beta model
    The inflated beta model is a generalized linear model with the log-likelihood 
    function defined as

    .. math::
        \log \mathcal{l}{\beta, \pi, \varphi, \theta \mid X,y,y_L,y_U}
        &= \sum_{y_j \leq y_L} \log ( \pi (1 - \varphi)) + \sum_{y_j \geq y_U} \log ( \pi \varphi ) \\
        &+ \sum_{y_L \leq y_j \leq y_U} \left(
            \log \frac{(1 - \pi) \, \Gamma (\theta)}{\Gamma (\theta \cdot S(X_j \beta)) \,
                \Gamma(\theta \cdot (1 - S(X_j \beta)))}
        + \log ((\theta \cdot S(X_j \beta) - 1) \, \log (1 - y_j))
        + \log ((\theta \cdot (1 - S(X_j \beta)) - 1) \, \log (1 - y_j))
        \right)

    where :math: `y_L` is the lower bound, :math: `y_U` is the upper bound, 
    :math: `\Gamma` is the Euler gamma function and :math:`S` is the sigmoid function
    """
    def __init__(self, verbose=False):
        self.add_params_init = [0.2,0.3,2.0]
        self.verbose = verbose        

    def sigmoid(self, x, params):
        z = np.dot(x, params)
        return 1.0/(1.0 + np.exp(-z))

    def predict(self, params, x, indep_var_list):
        """Function to generate predicted values for a given data

        Args:
            params (list): List of parameter estimates. Including pie, kesai and phi
            x (numpy.array): Matrix of independent variable values
            indep_var_list (list): List of independent variable names

        Returns:
            numpy.array: Predicted values using the model estimates
        """
        pie = params[-3]
        kesai = params[-2]
        phi = params[-1]
        mu = self.sigmoid(x, params[0:len(indep_var_list)+1].reshape(1, len(indep_var_list)+1).T)
        return pie * kesai + ((1 - mu) * phi)
    
    def log_likelihood(self, params):
        """Function to generate the Negative Log Likelihood Value for the given functional form

        Args:
            params (numpy.array): List of parameter values

        Returns:
            float: Negative Log Likelihood Value
        """
        nll_lr_term1 = 0
        nll_lr_term2 = 0
        nll_ur_term1 = 0
        nll_ur_term2 = 0
        nll_wr_term1 = 0
        nll_wr_term2 = 0
        nll_wr_term3 = 0
        nll_wr_term4 = 0
        nll_wr_term5 = 0
        nll_wr_term6 = 0
        
        pie = params[-3]
        kesai = params[-2]
        phi = params[-1]

        mu = self.sigmoid(self.x, params[0:len(self.indep_var_list)+1].reshape(1, len(self.indep_var_list)+1).T)

        slice_y = self.y[list(np.where(self.DWR == 1)[0])].copy()
        slice_mu = mu[list(np.where(self.DWR == 1)[0])].copy()

        nll_lr_term1 = np.log(pie)
        nll_lr_term2 = np.log(1 - kesai)

        nll_lr = np.sum((nll_lr_term1 + nll_lr_term2)*self.DLR[self.DLR==1])

        nll_ur_term1 = np.log(pie)
        nll_ur_term2 = np.log(kesai)

        nll_ur = np.sum((nll_ur_term1 + nll_ur_term2)*self.DUR[self.DUR==1])

        nll_wr_term1 = np.sum(np.log(1 - pie)*self.DWR[self.DWR==1])
        nll_wr_term2 = np.sum(sp.loggamma(phi)*self.DWR[self.DWR==1])
        nll_wr_term3 = -1 * np.sum(sp.loggamma(phi * slice_mu))
        nll_wr_term4 = -1 * np.sum(sp.loggamma(phi * (1 - slice_mu)))
        nll_wr_term5 = np.dot(((slice_mu*phi - 1)).T, np.log(slice_y))
        nll_wr_term6 = np.dot((((1 - slice_mu)*phi - 1)).T, np.log(1 - slice_y))

        nll_wr = nll_wr_term1 + nll_wr_term2 + nll_wr_term3 + nll_wr_term4 + nll_wr_term5 + nll_wr_term6
        
        nll = -1 * np.sum([nll_wr, nll_ur, nll_lr])

        return nll[0]
    
    def data_prep(self, df, indep_var_list, y_var, y_lower_bound, y_upper_bound):
        """Prepare the input data for the model fitting

        Args:
            df (pandas.DataFrame): The raw input data
            indep_var_list (list): List of independent variables
            y_var (str): Name of the dependent variable
            y_lower_bound (float): Lower bound for the dependent variable
            y_upper_bound (float): Upper bound for the dependent variable
        """
        self.indep_var_list = indep_var_list.copy()
        self.x = np.hstack((np.ones(df.shape[0]).reshape(df.shape[0],1), np.array(df[indep_var_list])))
        self.y = np.array(df[y_var])
        self.DWR = np.where((self.y > y_lower_bound) & (self.y < y_upper_bound),1,0)
        self.DUR = np.where(self.y >= y_upper_bound,1,0)
        self.DLR = np.where(self.y <= y_lower_bound,1,0)

    def gradient_(self, params):
        """Return the partial derivative of the log-likelihood function given the parameters

        Args:
            params (numpy.array): List of parameters required for the log-likelihood function

        Returns:
            numpy.array: Vectors of gradients
        """
        gradient = ndt.Gradient(self.log_likelihood)(params)
        return gradient
    
    def callback_function(self, params):
        """Call Back function to print iterations

        Args:
            params (numpy.array): List of parameters

        Returns:
            bool: True or False depending upon the change in Gradient of convergence
        """
        current_log_likelihood = self.log_likelihood(params)
        if self.verbose:
            print(f"Iteration {self.iter_}: Log-Likelihood = {current_log_likelihood}; Gradient = {np.linalg.norm(self.gradient_(params))}; Params = {params}")

        self.log_likelihood_values.append(current_log_likelihood)

        self.iter_ += 1
        return np.linalg.norm(self.gradient_(params)) <= self.tolerance
    
    def find_opt_params(self, init_params):
        """Optimization Function to get the optimal parameters

        Args:
            init_params (list): List of initial parameters

        Returns:
            scipy.optimize.minimize: The optimized resultant object
        """
        self.iter_ = 1
        self.log_likelihood_values = []
        self.tolerance = 1e-7
        
        init_params = np.array(init_params + self.add_params_init)
        result = minimize(self.log_likelihood, init_params, method = 'BFGS',
                          jac = self.gradient_, callback= self.callback_function, 
                          options={'disp': False})
        return result

    def estimate_params(self, init_params):
        """Function to generate the parameter estimate summary just like SAS

        Args:
            init_params (list): List of initial parameters

        Returns:
            pandas.DataFrame: Final summary of parameter estimates
        """
        result = self.find_opt_params(init_params)

        # Define the significance level (alpha)
        alpha = 0.05  # You can adjust the significance level as needed (e.g., 0.01 for 99% confidence intervals)

        # Degrees of freedom (typically, number of observations - number of parameters)
        df = self.x.shape[0]  # Adjust with the actual number of observations and parameters

        # Standard errors of parameter estimates
        standard_errors = np.sqrt(np.diag(result.hess_inv))

        # Calculate t-values
        t_values = result.x / standard_errors

        # Calculate two-tailed p-values
        p_values = 2 * (1 - t.cdf(np.abs(t_values), df))

        # Calculate critical t-value for the specified alpha and degrees of freedom
        critical_t_value = t.ppf(1 - alpha / 2, df)

        # Calculate 95% confidence intervals
        confidence_intervals = [
            (param - critical_t_value * se, param + critical_t_value * se)
            for param, se in zip(result.x, standard_errors)
        ]

        # Collate the results
        parameter_names = ["const", "Col1", "Col2", "Col3", "Col4", "pie", "kesai", "phi"]  # Replace with your parameter names
        all_metrics = list(zip(parameter_names, result.x, standard_errors, np.dot(df, np.ones(len(parameter_names))),
                                t_values, p_values, confidence_intervals, self.gradient_(result.x)))
        final_dict = {
            "Parameter": [],
            "Estimate": [],
            "Standard Error": [],
            "DF": [],
            "t Values": [],
            "Pr > |t|": [],
            "95% Confidence Limits": [],
            "Gradient": [],
        }
        for metric in all_metrics:
            for i, key__ in enumerate(final_dict.keys()):
                final_dict[key__].append(metric[i])

        return pd.DataFrame(final_dict)
