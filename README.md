# PyTorch Exercises

Small, self-contained Jupyter notebook covering manual optimization, time-series regression, and vectorized geometry with PyTorch.

## Exercises
- **Part 1 – Stochastic Gradient Descent:** Build logistic regression from scratch on synthetic data, hand-code SGD updates (no optimizer), and visualize the learned decision curve plus loss curve.
- **Part 2.1 – Bivariate Time Series:** Load `data.csv` (gappy/noisy measurements), normalize, train a 4-layer MLP to regress `(x(t), y(t))` with MSE loss, and plot reconstructed trajectories.
- **Part 2.2 – Feature Engineering:** Extend the time-series model with polynomial and sinusoidal features to capture periodic structure before feeding the MLP; compare training curves and predictions.
- **Part 3.1 – Distance Maps:** Vectorized nearest-distance map on a grid with no Python loops, then a batched variant returning `[B, res, res]` tensors for multiple point sets.
- **Part 3.2 – Scalable Distance Maps:** Chunked computation to handle very high resolutions (e.g., `res=16384`) without exhausting RAM.

## Run it
1) Install Python 3 deps: `pip install torch pandas matplotlib numpy`.
2) Download `data.csv` from the link in Part 2 markdown and place it beside `PyTorch.ipynb`.
3) Launch the notebook (`jupyter notebook PyTorch.ipynb` or run in VS Code/Colab) and execute cells in order.
4) Plots will show model fit, loss curves, and distance maps; the final chunked example can be slow but fits in memory.

CPU is enough for all sections; GPU is optional.
