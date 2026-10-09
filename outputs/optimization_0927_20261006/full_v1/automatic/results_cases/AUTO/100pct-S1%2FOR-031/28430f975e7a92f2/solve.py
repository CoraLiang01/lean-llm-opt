import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
if not set(['gen_per_lot', 'cost_per_lot']).issubset(energy_df.columns):
    raise KeyError('Missing required columns in energy.csv')
try:
    gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(int).to_dict()
    cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f'Error converting gen_per_lot or cost_per_lot to numeric: {e}')
for opt in option_ids:
    if opt not in gen_per_lot or opt not in cost_per_lot:
        raise KeyError(f'Missing coefficients for option {opt}')
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='demand')
m.optimize()