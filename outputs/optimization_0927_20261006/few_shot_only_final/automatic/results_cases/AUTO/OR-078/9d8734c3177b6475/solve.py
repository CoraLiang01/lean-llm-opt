import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
try:
    gen_per_lot = {}
    cost_per_lot = {}
    for (idx, row) in energy_df.iterrows():
        option = row['option']
        try:
            gen_per_lot[option] = float(row['gen_per_lot'])
        except Exception:
            raise ValueError(f"Invalid or missing gen_per_lot for option '{option}'")
        try:
            cost_per_lot[option] = float(row['cost_per_lot'])
        except Exception:
            raise ValueError(f"Invalid or missing cost_per_lot for option '{option}'")
except KeyError as e:
    raise KeyError(f'Missing required column in energy.csv: {e}')
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[option] * x_vars[option] for option in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[option] * x_vars[option] for option in option_ids)) >= 200, name='demand')
m.optimize()