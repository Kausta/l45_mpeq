## Note: need to have run the jupyter notebook first and saved final_messages with pickle

import numpy as np
import pandas as pd
import pickle as pkl
from pysr import PySRRegressor
import sympy
# import subprocess


N_messages_all = 2000
N_best = 5

MSG, ALGO_INPUT = 32, 11
VARIABLE_NAMES = ["p1", "p2", "s1", "s2", "a", "adj", "ph", "d1", "d2", "m1", "m2"]
IGNORE_VARIABLES = [] # [0, 1]

def criteria(algo_inputs):
    return None

take_only_nonzero, nz_thresh = False, 1e-2

file_name = 'bellman_ford_val_msgs.pkl'
with open(file_name, 'rb') as f:
    batched_msgs = pkl.load(f)

print(batched_msgs.shape)
assert batched_msgs.shape[1] == MSG + ALGO_INPUT

msgs = batched_msgs[:,:MSG]
algo_inputs = batched_msgs[:,MSG:]
print(msgs.shape, algo_inputs.shape)

mask = criteria(algo_inputs)
if mask is not None:
    print("Applying criteria")
    print(f"Keeping {np.sum(mask)} of {msgs.shape[0]}")
    msgs = msgs[mask]
    algo_inputs = algo_inputs[mask]

variables_to_keep = [x for x in range(ALGO_INPUT) if x not in IGNORE_VARIABLES]
VARIABLE_NAMES = [VARIABLE_NAMES[i] for i in variables_to_keep]
algo_inputs = algo_inputs[:, variables_to_keep]
ALGO_INPUT = len(variables_to_keep)

std_devs = np.array([np.std(msgs[:,i]) for i in range(msgs.shape[1])])
print(std_devs)
sort_idx = np.argsort(-std_devs)
best_N_idx = sort_idx[:N_best]
best_idx = np.argmax(std_devs)
print("Best msg index:", best_idx, ", std_dev:", std_devs[best_idx])
print(f"Best {N_best} msg indices:", best_N_idx, ", std_devs:", std_devs[best_N_idx])

def get_model():
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
            "heaviside(x) = x > 0.0f3 ? 1.0f3 : x < 0.0f3 ? 0.0f3 : 0.5f3",
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
        # maxdepth=5,
        multithreading=False
    )
    return model


print("All inputs")
# Largest std vs inputs
X_full = algo_inputs
y_full = msgs[:,best_N_idx[0]]

if take_only_nonzero:
    nonzero = np.abs(y_full) > nz_thresh
    X_full = X_full[nonzero]
    y_full = y_full[nonzero]

idxs = np.random.choice(np.arange(X_full.shape[0]), size=min(X_full.shape[0], N_messages_all), replace=False)

X = X_full[idxs,:]
y = y_full[idxs]

model = get_model()
model.fit(X, y, variable_names=VARIABLE_NAMES)
print(model)

# print(model.latex_table())
# subprocess.run("pbcopy", text=True, input=model.latex_table())
