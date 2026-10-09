import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
option_ids = energy_df['option'].tolist()
if 'gen_per_lot' not in energy_df.columns or 'cost_per_lot' not in energy_df.columns:
    raise KeyError("Missing required columns 'gen_per_lot' or 'cost_per_lot' in energy.csv")
try:
    gen_per_lot_dict = dict(zip(option_ids, energy_df['gen_per_lot'].astype(int)))
    cost_per_lot_dict = dict(zip(option_ids, energy_df['cost_per_lot'].astype(float)))
except Exception as e:
    raise ValueError(f"Error converting 'gen_per_lot' or 'cost_per_lot' to numeric: {e}")
if set(gen_per_lot_dict.keys()) != set(option_ids) or set(cost_per_lot_dict.keys()) != set(option_ids):
    raise ValueError('Mismatch in option IDs and parameter dictionaries.')
m = gp.Model('Electricity_Procurement_Lot_Selection')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_dict[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot_dict[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='demand')
m.optimize()