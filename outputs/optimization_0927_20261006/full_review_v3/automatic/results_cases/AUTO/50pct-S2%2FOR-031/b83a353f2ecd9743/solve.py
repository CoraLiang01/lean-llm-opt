import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
option_ids = energy_df['option'].tolist()
if 'cost_per_lot' not in energy_df.columns or 'gen_per_lot' not in energy_df.columns:
    raise KeyError("Missing required columns 'cost_per_lot' or 'gen_per_lot' in energy.csv")
try:
    cost_per_lot = {row['option']: float(row['cost_per_lot']) for (_, row) in energy_df.iterrows()}
    gen_per_lot = {row['option']: int(row['gen_per_lot']) for (_, row) in energy_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error converting cost_per_lot or gen_per_lot to numeric: {e}')
for oid in option_ids:
    if oid not in cost_per_lot or oid not in gen_per_lot:
        raise ValueError(f"Missing cost_per_lot or gen_per_lot for option '{oid}'")
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[oid] * x_vars[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[oid] * x_vars[oid] for oid in option_ids)) >= total_demand, name='DemandSatisfaction')
m.optimize()