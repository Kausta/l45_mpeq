## Note: need to have run the jupyter notebook first and saved final_messages with pickle

import numpy as np
import pandas as pd
import pickle as pkl
from pysr import PySRRegressor
# import subprocess


final_messages = pkl.load(open('final_messages.pkl', 'rb'))
# final_messages = pkl.load(open('third_party/symbolic_deep_learning/final_messages.pkl', 'rb'))

best_messages = np.argsort([np.std(final_messages['e%d'%(i,)]) for i in range(100)])
best_message = best_messages[-1]
print('best message index:', best_message)

print(final_messages[['e%d'%(best_message,), 'dx', 'dy', 'r', 'm1', 'm2']])


model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=100,
    binary_operators=["plus", "sub", "mult", "div", "greater", "pow"],
    unary_operators=["exp", "log"],
    extra_sympy_mappings={"if": lambda x, y, z: y if x else z},
    complexity_of_operators={"pow":3, "exp":3, "log":3},
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
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
