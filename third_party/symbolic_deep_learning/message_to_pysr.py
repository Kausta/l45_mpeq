## Note: need to have run the jupyter notebook first and saved final_messages with pickle

import numpy as np
import pandas as pd
import pickle as pkl
from pysr import PySRRegressor
import sympy
# import subprocess


final_messages = pkl.load(open('final_messages.pkl', 'rb'))
# final_messages = pkl.load(open('third_party/symbolic_deep_learning/final_messages.pkl', 'rb'))

best_messages = np.argsort([np.std(final_messages['e%d'%(i,)]) for i in range(100)])
best_message = best_messages[-1]
print('best message index:', best_message)

print(final_messages[['e%d'%(best_message,), 'dx', 'dy', 'r', 'm1', 'm2']])


model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=500,
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
    # maxsize=10,
    # maxdepth=5
)

X_full = final_messages[['dx', 'dy', 'r', 'm1', 'm2']].to_numpy()
y_full = final_messages[['e%d'%(best_message,)]].to_numpy().flatten()

idxs = np.random.choice(np.arange(X_full.shape[0]), size=5000, replace=False)

X = X_full[idxs,:]
y = y_full[idxs]

model.fit(X, y)

print(model)
# print(model.latex_table())
# subprocess.run("pbcopy", text=True, input=model.latex_table())
