import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
energy_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv', dtype=str, keep_default_na=False)
for col in ['gen_per_lot', 'cost_per_lot']:
    energy_df[col] = energy_df[col].astype(float)
option_ids = energy_df['option'].tolist()
gen_per_lot = dict(zip(energy_df['option'], energy_df['gen_per_lot']))
cost_per_lot = dict(zip(energy_df['option'], energy_df['cost_per_lot']))
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in option identifiers between data and parameter dictionaries.')
m = Model('electricity_procurement')
x_vars = m.addVars(option_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((cost_per_lot[i] * x_vars[i] for i in option_ids)), GRB.MINIMIZE)
m.addConstr(quicksum((gen_per_lot[i] * x_vars[i] for i in option_ids)) >= 200, name='demand')
m.optimize()