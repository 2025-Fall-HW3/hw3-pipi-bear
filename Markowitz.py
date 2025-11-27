"""
Package Import
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

start = "2019-01-01"
end = "2024-04-01"

# Initialize df and df_returns
df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust=False)
    df[asset] = raw["Adj Close"]

df_returns = df.pct_change().fillna(0)


"""
Problem 1: 

Implement an equal weighting strategy as dataframe "eqw". Please do "not" include SPY.
"""


class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 1 Below
        """
        # Assign equal weights to all non-excluded assets across all dates
        num_assets = len(assets)
        if num_assets > 0:
            equal_weight = 1.0 / num_assets
            # assign equal weight to all rows (dates) for the non-excluded assets
            self.portfolio_weights.loc[:, assets] = equal_weight
        # Ensure excluded asset has zero allocation
        self.portfolio_weights[self.exclude] = 0

        """
        TODO: Complete Task 1 Above
        """
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Problem 2:

Implement a risk parity strategy as dataframe "rp". Please do "not" include SPY.
"""


class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 2 Below
        """

        # For each date i, we calculate the std of the returns for the previous lookback days, then calculate the weight based on the definition stated in the spec
        # note: under this situation, the dates that cannot get the past lookback days will not have their weights assigned
        for i in range(self.lookback + 1, len(df)):
            # get the returns for the past lookback days (we're taking rows from i - lookback to i)
            # explain: window_returns is of shape (50, 11), which means the returns for the past 50 days (rows) for 11 assets (columns), use debugger to see the data
            window_returns = df_returns.copy()[assets].iloc[i - self.lookback : i]
            sigma = window_returns.std()

            # Inverse volatility weights, if the std is 0, we replace it with nan, then fill the nan with 0
            inv_sigma = 1.0 / sigma.replace(0, np.nan)
            inv_sigma = inv_sigma.fillna(0.0)

            # if the sum of the inverse volatility weights is 0, to avoid division by zero, we assign equal weights to all assets
            if inv_sigma.sum() == 0:
                weights = np.repeat(1.0 / len(assets), len(assets))
            else:
                weights = (inv_sigma / inv_sigma.sum()).values

            self.portfolio_weights.loc[df.index[i], assets] = weights

        # Ensure excluded asset has zero allocation
        self.portfolio_weights[self.exclude] = 0

        """
        TODO: Complete Task 2 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Problem 3:

Implement a Markowitz strategy as dataframe "mv". Please do "not" include SPY.
"""


class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(self.lookback + 1, len(df)):
            R_n = df_returns.copy()[assets].iloc[i - self.lookback : i]
            self.portfolio_weights.loc[df.index[i], assets] = self.mv_opt(
                R_n, self.gamma
            )

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                """
                TODO: Complete Task 3 Below

                Note: For detailed explanation including the basics about gurobi, refer to README.md.
                """

                w = model.addMVar(n, name="w", ub=1.0)

                model.addConstr(w.sum() == 1.0, name="budget")

                # Mean-variance objective:
                # maximize w^T mu - (gamma/2) * \frac{gamma}{2} w^T Sigma w
                mean_term = mu @ w
                risk_term = w @ (Sigma @ w)
                objective = mean_term - (gamma / 2.0) * risk_term
                model.setObjective(objective, gp.GRB.MAXIMIZE)

                """
                TODO: Complete Task 3 Above
                """
                model.optimize()

                # Check if the status is INF_OR_UNBD (code 4)
                if model.status == gp.GRB.INF_OR_UNBD:
                    print(
                        "Model status is INF_OR_UNBD. Reoptimizing with DualReductions set to 0."
                    )
                elif model.status == gp.GRB.INFEASIBLE:
                    # Handle infeasible model
                    print("Model is infeasible.")
                elif model.status == gp.GRB.INF_OR_UNBD:
                    # Handle infeasible or unbounded model
                    print("Model is infeasible or unbounded.")

                if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                    # Extract the solution
                    solution = []
                    for i in range(n):
                        var = model.getVarByName(f"w[{i}]")
                        # print(f"w {i} = {var.X}")
                        solution.append(var.X)

        return solution

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 1"
    )
    """
    NOTE: For Assignment Judge
    """
    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()

    # All grading logic is protected in grader.py
    judge.run_grading(args)
