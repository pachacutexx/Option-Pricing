# Option-Pricing
Option-Pricing is a comprehensive Python library for pricing options using various methods including the Binomial Tree, Trinomial Tree, and Black-Scholes model. 
Suitable for both educational purposes and practical applications, it aims to expand to include American options and other advanced pricing techniques.

## Features
- Binomial Tree method for European options
- Trinomial Tree method for European options
- Black-Scholes formula implementation
- (Upcoming features)

## Installation

git clone https://github.com/pachacutexx/Option-Pricing.git

cd Option-Pricing/src/

## Usage

Below are some examples of how to use the 'EuoropeanOption' class to price European options using different methods.

Pricing a European call option using a binomial tree

from EuropeanOption import EuropeanOption

Initialize a European call option

call_option = EuropeanOption(S0=100, K=100, T=1, r=0.05, sigma=0.2, is_call=True, steps=252)

Calculate the price using a binomial tree

binomial_call_price = call_option.price(method='binomial')

print(f"Call option price using Binomial Tree: {binomial_call_price:.2f}")

## Contributing
While this project is primarily for personal educational purposes, suggestions and improvements are welcome. 
Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details
