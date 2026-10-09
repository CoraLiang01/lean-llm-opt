import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
try:
    cost_per_lot = pd.to_numeric(energy_df['cost_per_lot'], errors='raise')
except Exception as e:
    raise ValueError(f'Failed to convert cost_per_lot to numeric: {e}')
cost_per_lot_dict = dict(zip(energy_df['option'], cost_per_lot))
try:
    gen_per_lot = pd.to_numeric(energy_df['gen_per_lot'], errors='raise')
except Exception as e:
    raise ValueError(f'Failed to convert gen_per_lot to numeric: {e}')
gen_per_lot_dict = dict(zip(energy_df['option'], gen_per_lot))
m = gp.Model('Electricity_Procurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_dict[i] * x_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot_dict[i] * x_vars[i] for i in option_ids)) >= 200, name='demand')
m.optimize()