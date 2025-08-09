import numpy as np
from powerMethod import power_method_for_polynomial

def main():
    polynomials = {
        "x^3 + x^2 - 5x + 3 = 0": [3, -5, 1, 1],
        "x^3 - 3x - 1 = 0": [-1, -3, 0, 1],
        "x^8 + 101x^7 + 208.01x^6 + 10891.01x^5 + 9802.08x^4 + 79108.9x^3 - 99902x^2 + 790x - 1000 = 0": 
            [-1000, 790, -99902, 79108.9, 9802.08, 10891.01, 208.01, 101, 1]
    }

    for equation, coeffs in polynomials.items():
        eigenvalue, _ = power_method_for_polynomial(coeffs)
        print(f"方程: {equation}")
        print(f"模最大的根: {eigenvalue:.6f}\n")

if __name__ == "__main__":
    main()