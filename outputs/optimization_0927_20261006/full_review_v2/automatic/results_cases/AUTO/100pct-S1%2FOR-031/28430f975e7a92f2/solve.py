import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].astype(str).tolist()
try:
    gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'gen_per_lot' to int: {e}")
try:
    cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'cost_per_lot' to float: {e}")
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in option identifiers between index set and parameter dictionaries.')
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='demand')
m.optimize()