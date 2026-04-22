import numpy as np
import matplotlib.pyplot as plt

# User-provided parameters
a = 0.135
b = 54.0
d_values = [0.1, 0.2, 0.308, 0.5, 0.8, 1.2, 1.6, 2.0]

# Threshold is at x = b/a
threshold = b / a

# Choose a range around threshold so the effect is visible
x = np.linspace(300, 500, 1200)

def transfer_fn(x, a, b, d):
    y = a * x - b
    out = np.empty_like(y, dtype=float)
    small = np.isclose(y, 0.0, atol=1e-10)
    out[small] = 1.0 / d
    out[~small] = y[~small] / (1 - np.exp(-d * y[~small]))
    return out

plt.figure(figsize=(9, 5.5))
for d in d_values:
    plt.plot(x, transfer_fn(x, a, b, d), label=f"d = {d}")

plt.axvline(threshold, linestyle="--", linewidth=1, label=f"threshold = {threshold:.0f}")
plt.xlabel("input_current")
plt.ylabel(r"$(a x - b)/(1-\exp(-d(a x-b)))$")
plt.title("Effect of varying d with a = 0.135 and b = 54")
plt.legend()
plt.tight_layout()
plt.show()