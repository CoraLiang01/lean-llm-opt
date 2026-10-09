import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
option_ids = energy_df['option'].tolist()
if 'gen_per_lot' not in energy_df.columns or 'cost_per_lot' not in energy_df.columns:
    raise KeyError("Missing required columns 'gen_per_lot' or 'cost_per_lot' in energy.csv")
try:
    gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(int).to_dict()
    cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'gen_per_lot' or 'cost_per_lot' to numeric: {e}")
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch between option IDs and parameter keys in energy.csv')
m = gp.Model('Electricity_Lot_Procurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= total_demand, name='DemandSatisfaction')
m.optimize()