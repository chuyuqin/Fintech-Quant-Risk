"""Generalized Black–Scholes and exact course binomial dividend recursion."""

import numpy as np
from scipy.stats import norm
from numba import njit


def gbsm(call, s, k, t, r, b, v):
    z = 1 if call else -1
    d1 = (np.log(s / k) + (b + v * v / 2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    carry = np.exp((b - r) * t)
    disc = np.exp(-r * t)
    value = z * (s * carry * norm.cdf(z * d1) - k * disc * norm.cdf(z * d2))
    delta = z * carry * norm.cdf(z * d1)
    gamma = norm.pdf(d1) * carry / (s * v * np.sqrt(t))
    vega = s * carry * norm.pdf(d1) * np.sqrt(t)
    rho = z * t * k * disc * norm.cdf(z * d2)
    theta = (
        -s * carry * norm.pdf(d1) * v / (2 * np.sqrt(t))
        - z * (b - r) * s * carry * norm.cdf(z * d1)
        - z * r * k * disc * norm.cdf(z * d2)
    )
    return np.array([value, delta, gamma, vega, rho, theta])


@njit
def american(call, s, k, t, r, b, v, n):
    dt = t / n
    u = np.exp(v * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(b * dt) - d) / (u - d)
    df = np.exp(-r * dt)
    z = 1 if call else -1
    vals = np.empty(n + 1)
    for i in range(n + 1):
        vals[i] = max(0.0, z * (s * u**i * d ** (n - i) - k))
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            vals[i] = max(
                z * (s * u**i * d ** (j - i) - k),
                df * (p * vals[i + 1] + (1 - p) * vals[i]),
            )
    return vals[0]


@njit
def american_discrete(call, s, k, t, r, amounts, times, v, n):
    if len(times) == 0 or times[0] > n:
        return american(call, s, k, t, r, r, v, n)
    dt = t / n
    u = np.exp(v * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)
    z = 1 if call else -1
    m = times[0]
    vals = np.empty(m + 1)
    for i in range(m + 1):
        price = s * u**i * d ** (m - i)
        continuation = american_discrete(
            call,
            price - amounts[0],
            k,
            t - m * dt,
            r,
            amounts[1:],
            times[1:] - m,
            v,
            n - m,
        )
        vals[i] = max(continuation, z * (price - k), 0.0)
    for j in range(m - 1, -1, -1):
        for i in range(j + 1):
            vals[i] = max(
                z * (s * u**i * d ** (j - i) - k),
                0.0,
                df * (p * vals[i + 1] + (1 - p) * vals[i]),
            )
    return vals[0]


def american_greeks(call, s, k, t, r, b, v, n=500):
    parms = np.array([s, k, t, r, b, v])
    f = lambda p: american(call, *p, n)
    value = f(parms)
    grad = []
    # FiniteDiff central gradient; nonsmooth exercise boundaries require both sides.
    for j in range(6):
        h = np.cbrt(np.finfo(float).eps) * max(abs(parms[j]), 1)
        p = parms.copy()
        p[j] += h
        q = parms.copy()
        q[j] -= h
        grad.append((f(p) - f(q)) / (2 * h))
    p = parms.copy()
    p[0] += 1.5
    q = parms.copy()
    q[0] -= 1.5
    gamma = (f(p) + f(q) - 2 * value) / 1.5**2
    return np.array([value, grad[0], gamma, grad[5], grad[3], grad[2]])
