import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
try:
    gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(int).to_dict()
    cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f'Error converting numeric columns: {e}')
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch between option IDs and parameter keys in energy.csv.')
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x_vars[i] for i in option_ids)) >= 200, name='demand')
m.optimize()