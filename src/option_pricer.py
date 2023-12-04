import numpy as np
from scipy.stats import norm

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

    def trinomial_price(self):
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(2 * dt))
        d = 1 / u
        m = 1
        pu = ((np.exp(self.r * dt / 2) - np.exp(-self.sigma * np.sqrt(dt / 2))) /
              (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2))))**2
        pd = ((np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(self.r * dt / 2)) /
              (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2))))**2
        pm = 1 - pu - pd

        # Initialize asset prices at maturity
        ST = np.zeros(2 * self.steps + 1)
        ST[0] = self.S0 * d**self.steps

        for i in range(1, 2 * self.steps + 1):
            ST[i] = ST[i - 1] * u if i % 2 == 1 else ST[i - 1] * m

        # Calculate option values at maturity
        option_values = np.maximum(ST - self.K, 0) if self.is_call else np.maximum(self.K - ST, 0)

        # Step backwards through the tree
        for i in range(self.steps, 0, -1):
            for j in range(2 * i - 1):
                option_values[j] = (pu * option_values[j + 2] + pm * option_values[j + 1] + pd * option_values[j]) * np.exp(-self.r * dt)

        return option_values[0]

    def black_scholes_price(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.is_call:
            price = (self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1))

        return price

    def price(self, method='binomial'):
        if method == 'binomial':
            return self.binomial_price()
        elif method == 'trinomial':
            return self.trinomial_price()
        elif method == 'black_scholes':
            return self.black_scholes_price()
        else:
            raise ValueError("Invalid method. Choose 'binomial', 'trinomial', or 'black_scholes'.")

