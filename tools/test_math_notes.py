#!/usr/bin/env python3
"""Numeric smoke tests for the most error-prone formulas in the notes."""

from __future__ import annotations

import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def close(actual: float, expected: float, tol: float = 1e-8) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"expected {expected}, got {actual}")


def test_lse_softmax() -> None:
    s = [-1.0, 0.3, 2.0]
    denominator = sum(math.exp(x) for x in s)
    softmax = [math.exp(x) / denominator for x in s]
    h = 1e-6
    for i in range(len(s)):
        plus = s.copy()
        minus = s.copy()
        plus[i] += h
        minus[i] -= h
        lse_plus = math.log(sum(math.exp(x) for x in plus))
        lse_minus = math.log(sum(math.exp(x) for x in minus))
        close((lse_plus - lse_minus) / (2 * h), softmax[i], 1e-7)


def test_strong_convex_contraction() -> None:
    L, mu = 10.0, 2.0
    cases = [
        (1 / L, 1 - mu / L),
        (2 / (L + mu), (L - mu) / (L + mu)),
    ]
    for gamma, expected in cases:
        actual = max(abs(1 - gamma * eigenvalue) for eigenvalue in (mu, L))
        close(actual, expected)


def test_sgd_noise_floor() -> None:
    mu, sigma, gamma = 2.0, 0.4, 0.05
    a = 1 - gamma * mu
    variance = 0.0
    for _ in range(10000):
        variance = a * a * variance + gamma * gamma * sigma * sigma
    expected = gamma * sigma**2 / (2 * mu - gamma * mu**2)
    close(variance, expected, 1e-11)


def test_adafactor_example() -> None:
    G = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    r = [sum(x * x for x in row) / len(row) for row in G]
    c = [
        sum(G[i][j] ** 2 for i in range(len(G))) / len(G)
        for j in range(len(G[0]))
    ]
    denominator = sum(r) / len(r)
    estimated_v11 = r[0] * c[0] / denominator
    close(estimated_v11, 34 / 13)


def test_sphere_estimator_and_ball_smoothing() -> None:
    # In one dimension the unit sphere is {-1, 1}; f=x^3/3 has a nonzero
    # third derivative, so confusing sphere smoothing with ball smoothing
    # would produce a different answer.
    x, tau = 0.7, 0.2

    def f(value: float) -> float:
        return value**3 / 3

    sphere_estimator = sum(
        (f(x + tau * direction) - f(x)) * direction / tau
        for direction in (-1.0, 1.0)
    ) / 2
    ball_smoothing_gradient = (f(x + tau) - f(x - tau)) / (2 * tau)
    close(sphere_estimator, ball_smoothing_gradient)


def test_mezo_central_difference() -> None:
    rng = random.Random(20261025)
    x = [0.3, -0.4, 0.8, 1.1, -0.6]
    mu, tau = 0.7, 1e-3
    samples = 100_000
    estimate = [0.0] * len(x)
    for _ in range(samples):
        u = [rng.gauss(0.0, 1.0) for _ in x]
        f_plus = 0.5 * mu * sum((xi + tau * ui) ** 2 for xi, ui in zip(x, u))
        f_minus = 0.5 * mu * sum((xi - tau * ui) ** 2 for xi, ui in zip(x, u))
        scale = (f_plus - f_minus) / (2 * tau)
        for i, ui in enumerate(u):
            estimate[i] += scale * ui / samples
    expected = [mu * xi for xi in x]
    for actual, target in zip(estimate, expected):
        close(actual, target, 0.03)


def test_projection_dimensions_and_notes() -> None:
    # G is 5x3 and P is 5x2. P^T G is 2x3; P(P^T G) is 5x3.
    G_shape = (5, 3)
    P_shape = (5, 2)
    projected_shape = (P_shape[1], G_shape[1])
    lifted_shape = (P_shape[0], projected_shape[1])
    assert projected_shape == (2, 3)
    assert lifted_shape == G_shape

    note = (ROOT / "notes/Lecture-10_Subspace-Optimizers.md").read_text(
        encoding="utf-8"
    )
    assert r"P\tilde G" in note
    if "不能把未经过投影的" not in note:
        raise AssertionError("missing explanation for PG dimension mismatch")


def test_content_regressions() -> None:
    l5 = (ROOT / "notes/Lecture-05_Gradient-Descent.md").read_text(
        encoding="utf-8"
    )
    assert r"r(x)=G(x)-x" in l5
    assert r"\sum_i\alpha_{k,i}^\star G(x_{k-i})" in l5
    assert r"f^\star=\inf_x f(x)>-\infty" in l5

    l6 = (ROOT / "notes/Lecture-06_Stochastic-Gradient-Descent.md").read_text(
        encoding="utf-8"
    )
    assert r"\frac{\gamma\sigma^2}{2\mu-\gamma\mu^2}" in l6

    l7 = (ROOT / "notes/Lecture-07_Transformer-and-LLM-I.md").read_text(
        encoding="utf-8"
    )
    assert r"Q_i=XW_Q^{(i)}" in l7
    assert r"\mathrm{Attn}(Q_i,K_i,V_i)" in l7
    assert r"\mathrm{Attn}(QW_Q" not in l7

    l11 = (ROOT / "notes/Lecture-11_Zeroth-Order-Optimizers.md").read_text(
        encoding="utf-8"
    )
    assert r"\mathrm{Unif}(B_d)" in l11
    assert r"\frac{f_+-f_-}{2\tau}u" in l11

    l12 = (ROOT / "notes/Lecture-12_Distributed-Optimizers.md").read_text(
        encoding="utf-8"
    )
    assert "| 2 | micro-batch 2 | micro-batch 1 |" in l12
    assert "Random-$k$" in l12

    l13 = (ROOT / "notes/Lecture-13_Mixed-Precision-Training.md").read_text(
        encoding="utf-8"
    )
    assert "BF16 与 FP32 具有同样宽的指数范围" in l13

    l14 = (ROOT / "notes/Lecture-14_Efficient-Fine-Tuning-and-Inference.md").read_text(
        encoding="utf-8"
    )
    assert "**可训练参数**" in l14

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "课程可以分成六个阶段" in readme


def main() -> None:
    tests = [
        test_lse_softmax,
        test_strong_convex_contraction,
        test_sgd_noise_floor,
        test_adafactor_example,
        test_sphere_estimator_and_ball_smoothing,
        test_mezo_central_difference,
        test_projection_dimensions_and_notes,
        test_content_regressions,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"OK: {len(tests)} numerical/content tests")


if __name__ == "__main__":
    main()

