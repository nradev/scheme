# Maximum Sharpe Ratio

# We load the S&P 500 dataset composed of the daily prices of 18
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28.

from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import InverseVolatility, MeanRisk, ObjectiveFunction
from skfolio.portfolio import Portfolio
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()

# Select 18 specific assets for this example
selected_assets = [
    "AAPL",
    "AMD",
    "BAC",
    "BBY",
    "CVX",
    "GE",
    "HD",
    "JNJ",
    "JPM",
    "LLY",
    "MRK",
    "MSFT",
    "PEP",
    "PFE",
    "PG",
    "UNH",
    "WMT",
    "XOM",
]
prices = prices[selected_assets]

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# We create a Maximum Sharpe Ratio model and then fit it on the training set.
model = MeanRisk(
    risk_measure=RiskMeasure.STANDARD_DEVIATION,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    portfolio_params=dict(name="Max Sharpe"),
    save_problem=True,
)
model.fit(X_train)

# To compare this model, we use an inverse volatility benchmark using
# the InverseVolatility estimator:
benchmark = InverseVolatility(portfolio_params=dict(name="Inverse Vol"))
benchmark.fit(X_train)

# We evaluate the fitted portfolios on the test set:
pred_model = Portfolio(weights=model.weights_, X=X_test, name="Max Sharpe")
pred_bench = Portfolio(weights=benchmark.weights_, X=X_test, name="Inverse Vol")

# For improved analysis, we load both predicted portfolios into a skfolio.population.Population:
population = Population([pred_model, pred_bench])


portfolio_composition = population.composition()
print("\nPortfolio Composition:\n")
print(portfolio_composition)

summary_table = population.summary()
print("\nComparison of Test Set Performance:\n")
print(summary_table)
