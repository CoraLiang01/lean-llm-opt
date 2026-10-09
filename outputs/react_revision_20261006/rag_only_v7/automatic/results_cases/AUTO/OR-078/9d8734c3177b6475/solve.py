import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
energy_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
if len(option_ids) != len(set(option_ids)):
    raise ValueError('Duplicate option identifiers found in energy.csv.')
try:
    gen_per_lot = pd.Series(energy_df['gen_per_lot'].astype(float).values, index=energy_df['option']).to_dict()
    cost_per_lot = pd.Series(energy_df['cost_per_lot'].astype(float).values, index=energy_df['option']).to_dict()
except Exception as e:
    raise ValueError(f'Error converting numeric fields in energy.csv: {e}')
demand = 200.0
for i in option_ids:
    if i not in gen_per_lot or i not in cost_per_lot:
        raise ValueError(f'Missing coefficients for option {i}.')
m = gp.Model('electricity_lot_sizing')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(option_ids, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x_vars[i] for i in option_ids)), GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x_vars[i] for i in option_ids)) >= demand, name='demand')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in option_ids:
        var = x_vars[i]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')