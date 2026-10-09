import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv', dtype=str, keep_default_na=False)
if df['Shift'].nunique() != 24 or len(df) != 24:
    raise ValueError('Expected exactly 24 unique shifts in the input data.')
shifts = df['Shift'].astype(int).tolist()
shift_set = set(shifts)
if shift_set != set(range(1, 25)):
    raise ValueError('Shifts must be exactly 1..24.')
required_staff = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))
m = Model('bus_staffing')
x_vars = m.addVars(shifts, vtype=GRB.INTEGER, lb=0, name='')
for s in shifts:
    covering_t = [(s - i - 1) % 24 + 1 for i in range(4)]
    m.addConstr(quicksum((x_vars[t] for t in covering_t)) >= required_staff[s], name=f'cover_{s}')
m.setObjective(quicksum((x_vars[t] for t in shifts)), GRB.MINIMIZE)
m.optimize()