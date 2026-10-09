import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].astype(str).tolist()
try:
    cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'cost_per_lot' to float for all options: {e}")
try:
    gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'gen_per_lot' to float for all options: {e}")
m = gp.Model('Electricity_Procurement_Lot_MIP')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for opt in option_ids:
        val = x_vars[opt].X
        if val > 1e-06:
            print(f"  {opt}: {int(round(val))} lots (tech: {energy_df.loc[energy_df['option'] == opt, 'tech'].values[0]}, gen_per_lot: {gen_per_lot[opt]}, cost_per_lot: {cost_per_lot[opt]})")
else:
    print(f'No optimal solution found. Status: {m.status}')