"""
Package Import
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
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

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust=False)
    Bdf[asset] = raw["Adj Close"]

# Two evaluation datasets:
# - sharpRatio_ge1_df: 2019–2024
# - better_than_SPYSharpRatio_df: 2012–2024
sharpRatio_ge1_df = Bdf.loc["2019-01-01":"2024-04-01"]
better_than_SPYSharpRatio_df = Bdf

# Preserve original names used by the grader
df = sharpRatio_ge1_df

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(
        self,
        price,
        exclude,
        lookback_momentum=5,
        lookback_volatility=6,
        gamma=0,
        alpha=0.7,
    ):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback_momentum = lookback_momentum
        self.lookback_volatility = lookback_volatility
        self.gamma = gamma
        self.alpha = alpha

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # For each rebalancing date, compute momentum (mean return) with its window,
        # volatility (std) with its own window,
        # build risk-adjusted positive momentum signal, and normalize to weights.
        warmup = max(self.lookback_momentum, self.lookback_volatility)
        for i in range(warmup + 1, len(self.price)):
            window_mom = self.returns.copy()[assets].iloc[
                i - self.lookback_momentum : i
            ]
            window_vol = self.returns.copy()[assets].iloc[
                i - self.lookback_volatility : i
            ]

            momentum = window_mom.mean()  # average daily return over mom window
            volatility = window_vol.std(ddof=1)  # sample std over vol window

            # Risk-adjusted momentum signal: max(0, momentum) / volatility
            positive_momentum = momentum.clip(lower=0.0)
            denom = volatility.replace(0, np.nan)
            signal = (positive_momentum / denom).fillna(0.0) ** self.alpha

            # Normalize signals to get portfolio weights; fallback to equal weight
            signal_sum = signal.sum()
            if signal_sum == 0:
                weights = np.repeat(1.0 / len(assets), len(assets))
            else:
                weights = (signal / signal_sum).values

            self.portfolio_weights.loc[self.price.index[i], assets] = weights

        # Ensure excluded asset has zero allocation
        self.portfolio_weights[self.exclude] = 0

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
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
    from grader_2 import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

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

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()

    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
