import numpy as np
import pickle as pkl
from pysr import PySRRegressor
import sympy


X = 0.5 + np.random.normal(size=(1000,1))
# y = np.array([x**2 if x > 0 else x/2 for x in list(X.flatten())])
y = np.abs(X).flatten() + 1e-2 * np.random.normal(size=X.shape[0])
# y = np.array([1.0 if x > 0.5 else 0.0 for x in list(X.flatten())]) + 1e-2 * np.random.normal(size=X.shape[0])

model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=100,
    binary_operators=[
        "plus", "sub", "mult", "div", "pow",
        "min", "max",
    ],
    unary_operators=[
        "exp", "log", "abs",
        "relu(x) = (x >= 0.0f3) ? x : 0.0f3",
        "heaviside(x) = x > 0.0f3 ? 1.0f3 : x == 0.0f3 ? 0.5f3 : 0.0f3",
    ],
    extra_sympy_mappings={
        "min": lambda x, y: sympy.Piecewise((x, x < y), (y, True)),
        "max": lambda x, y: sympy.Piecewise((y, x < y), (x, True)),
        "abs": lambda x: sympy.Piecewise((x, x >= 0.0), (-x, True)),
        "relu": lambda x: sympy.Piecewise((x, x >= 0.0), (0.0, True)),
        "heaviside": sympy.Heaviside,
    },
    complexity_of_operators={"pow":3, "exp":3, "log":3},
    # loss="loss(x, y) = (x - y)^2",
    loss="L1DistLoss()",
    maxsize=10,
    maxdepth=5
)

model.fit(X, y)
