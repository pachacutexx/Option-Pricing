import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

class EuropeanOption:
    def __init__(self, S0, K, T, r, sigma, is_call=True, steps=100):
        self.S0 = S0       # Underlying asset price
        self.K = K         # Strike price
        self.T = T         # Time to maturity
        self.r = r         # Risk-free rate
        self.sigma = sigma # Volatility
        self.is_call = is_call # Call or Put
        self.steps = steps # Number of steps in the binomial tree

    def binomial_price(self):
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)

        # Initialize asset prices at maturity
        ST = np.zeros(self.steps + 1)
        ST[0] = self.S0 * d**self.steps

        for j in range(1, self.steps + 1):
            ST[j] = ST[j - 1] * u / d

        # Calculate option values at maturity
        option_values = np.zeros(self.steps + 1)
        if self.is_call:
            option_values = np.maximum(ST - self.K, 0)
        else:
            option_values = np.maximum(self.K - ST, 0)

        # Step backwards through the tree
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j] = (p * option_values[j + 1] + (1 - p) * option_values[j]) * np.exp(-self.r * dt)

        return option_values[0]

    def _generate_stock_prices(self, steps, dt):
        up = np.exp(self.sigma * np.sqrt(2 * dt))
        down = 1 / up

        vec_u = np.cumprod(up * np.ones(steps))
        vec_d = np.cumprod(down * np.ones(steps))

        stock_prices = np.concatenate((vec_d[::-1], [1.0], vec_u)) * self.S0
        return stock_prices

    def trinomial_price(self):
        dt = self.T / self.steps
        discount = np.exp(-self.r * dt)

        # Calculate risk-neutral probabilities
        pu = (
            (np.exp(self.r * dt / 2) - np.exp(-self.sigma * np.sqrt(dt / 2))) /
            (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))
        ) ** 2
        pd = (
            (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(self.r * dt / 2)) /
            (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))
        ) ** 2
        pm = 1 - pu - pd

        # Generate stock prices at maturity
        s = self._generate_stock_prices(self.steps, dt)

        # Define Payoff
        final_payoff = np.maximum(s - self.K, 0) if self.is_call else np.maximum(self.K - s, 0)
        nxt_vec_prices = final_payoff

        # Iterations for the calculation of payoffs
        for i in range(1, self.steps + 1):
            vec_stock = self._generate_stock_prices(self.steps - i, dt)
            expectation = np.zeros(len(vec_stock))

            for j in range(len(expectation)):
                tmp = nxt_vec_prices[j] * pd
                tmp += nxt_vec_prices[j + 1] * pm
                tmp += nxt_vec_prices[j + 2] * pu

                expectation[j] = tmp
            # Discount option payoff
            nxt_vec_prices = discount * expectation

        # Return the expected discounted value of the option at t=0
        return nxt_vec_prices[0]

    def black_scholes_price(self):
        # Calculate d1 using the Black-Scholes formula components.
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
        # Calculate d2, which is derived from d1 but adjusted by the volatility over time.
        d2 = d1 - self.sigma * np.sqrt(self.T)
    
        # Calculate the option price based on whether it is a call or a put option.
        if self.is_call:
            price = (self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1))
    
        # Return the calculated price.
        return price


    def price(self, method='black_scholes'):
        if method == 'binomial':
            return self.binomial_price()
        elif method == 'trinomial':
            return self.trinomial_price()
        elif method == 'black_scholes':
            return self.black_scholes_price()
        else:
            raise ValueError("Invalid method. Choose 'binomial', 'trinomial', or 'black_scholes'.")

class AmericanOption:
    def __init__(self, S0, K, T, r, sigma, is_call=True, steps=100):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.is_call = is_call
        self.steps = steps

    def binomial_price(self):
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        discount = np.exp(-self.r * dt)

        # Initialize asset prices at maturity
        ST = np.zeros(self.steps + 1)
        ST[0] = self.S0 * d**self.steps

        for j in range(1, self.steps + 1):
            ST[j] = ST[j - 1] * u / d

        # Calculate option values at maturity
        option_values = np.zeros(self.steps + 1)
        for j in range(self.steps + 1):
            if self.is_call:
                option_values[j] = max(ST[j] - self.K, 0)
            else:
                option_values[j] = max(self.K - ST[j], 0)

        # Step backwards through the tree
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j] = max(
                    (p * option_values[j + 1] + (1 - p) * option_values[j]) * discount,
                    ST[j] - self.K if self.is_call else self.K - ST[j]
                )

        return option_values[0]

def _generate_stock_prices(self, steps, dt):
        up = np.exp(self.sigma * np.sqrt(2 * dt))
        down = 1 / up

        vec_u = np.cumprod(up * np.ones(steps))
        vec_d = np.cumprod(down * np.ones(steps))

        stock_prices = np.concatenate((vec_d[::-1], [1.0], vec_u)) * self.S0
        return stock_prices

def trinomial_price(self):
        dt = self.T / self.steps
        discount = np.exp(-self.r * dt)

        # Calculate risk-neutral probabilities
        pu = (
            (np.exp(self.r * dt / 2) - np.exp(-self.sigma * np.sqrt(dt / 2))) /
            (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))
        ) ** 2
        pd = (
            (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(self.r * dt / 2)) /
            (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))
        ) ** 2
        pm = 1 - pu - pd

        # Generate stock prices at maturity
        s = self._generate_stock_prices(self.steps, dt)

        # Initialize option values at maturity
        option_values = np.maximum(s - self.K, 0) if self.is_call else np.maximum(self.K - s, 0)

        # Iterations for the calculation of option values
        for i in range(self.steps - 1, -1, -1):
            vec_stock = self._generate_stock_prices(i, dt)
            new_option_values = np.zeros(len(vec_stock))

            for j in range(len(vec_stock)):
                holding_value = discount * (pu * option_values[j + 2] + pm * option_values[j + 1] + pd * option_values[j])
                exercise_value = max(vec_stock[j] - self.K, 0) if self.is_call else max(self.K - vec_stock[j], 0)

                new_option_values[j] = max(holding_value, exercise_value)

            option_values = new_option_values

        # Return the option value at t=0
        return option_values[0]

def _simulate_asset_paths(self, n_paths, n_steps, dt, antithetic=True):
    """
    Simulate random paths for the underlying asset using Geometric Brownian Motion.
    If antithetic variates are used, the number of paths will be doubled.
    """
    if antithetic:
        total_paths = 2 * n_paths
    else:
        total_paths = n_paths

    paths = np.zeros((n_steps + 1, total_paths))
    paths[0] = self.S0

    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        if antithetic:
            z = np.concatenate((z, -z))
        paths[t] = paths[t - 1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)

    return paths


 def _longstaff_schwartz_payoff(self, paths, dt):
        """
        Calculate the payoff for each path using the Longstaff-Schwartz method with sklearn linear regression,
        including the previous price and its square as regression factors.
        """
        n_steps, n_paths = paths.shape
        payoffs = np.zeros(n_paths)
        values = np.zeros_like(paths)

        # Calculate payoffs at each time step
        if self.is_call:
            values = np.maximum(paths - self.K, 0)
        else:
            values = np.maximum(self.K - paths, 0)

        # Start from the second-to-last time step
        for t in range(n_steps - 2, 0, -1):
            itm = values[t] > 0
            regression_factors = np.vstack([
                np.ones(np.sum(itm)), 
                paths[t, itm], 
                paths[t, itm]**2
            ]).T
            regression_targets = values[t + 1, itm] * np.exp(-self.r * dt)
            reg = LinearRegression().fit(regression_factors, regression_targets)
            continuation_values = reg.predict(regression_factors)

            exercise = itm & (values[t] > continuation_values)
            payoffs[exercise] = values[t, exercise]
            payoffs[~exercise] = payoffs[~exercise] * np.exp(-self.r * dt)

        # Handle the last time step
        payoffs = np.where(values[1] > 0, values[1], payoffs * np.exp(-self.r * dt))
        return payoffs
     
    def lsmc_price(self, n_paths=10000, confidence_level=0.95):
        dt = self.T / self.steps
        paths = self._simulate_asset_paths(n_paths, self.steps, dt)
        payoffs = self._longstaff_schwartz_payoff(paths, dt)
        price = np.mean(payoffs)

        # Calculate the Z-score for the given confidence level
        z_score = norm.ppf((1 + confidence_level) / 2)

        # Confidence interval
        std_error = np.std(payoffs) / np.sqrt(n_paths)
        conf_interval = (price - z_score * std_error, price + z_score * std_error)

        return price, conf_interval

    def price(self, method='trinomial'):
        if method == 'binomial':
            return self.binomial_price()
        elif method == 'trinomial':
            return self.trinomial_price()
        elif method == 'lsmc':
            return self.lsmc_price()
        else:
            raise ValueError("Invalid method. Choose 'binomial', 'trinomial', or 'lsmc'.")
