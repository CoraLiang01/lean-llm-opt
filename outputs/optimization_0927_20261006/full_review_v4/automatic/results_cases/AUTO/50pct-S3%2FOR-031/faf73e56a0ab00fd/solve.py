import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
try:
    cost_per_lot = {}
    gen_per_lot = {}
    for (idx, row) in energy_df.iterrows():
        option = row['option']
        cost_str = row['cost_per_lot']
        if cost_str.strip() == '':
            raise ValueError(f"Missing cost_per_lot for option '{option}'")
        try:
            cost = float(cost_str)
        except Exception:
            raise ValueError(f"Invalid cost_per_lot '{cost_str}' for option '{option}'")
        cost_per_lot[option] = cost
        gen_str = row['gen_per_lot']
        if gen_str.strip() == '':
            raise ValueError(f"Missing gen_per_lot for option '{option}'")
        try:
            gen = float(gen_str)
        except Exception:
            raise ValueError(f"Invalid gen_per_lot '{gen_str}' for option '{option}'")
        gen_per_lot[option] = gen
    if set(cost_per_lot.keys()) != set(option_ids) or set(gen_per_lot.keys()) != set(option_ids):
        raise ValueError('Mismatch in option IDs between cost_per_lot and gen_per_lot.')
except Exception as e:
    raise RuntimeError(f'Error processing energy.csv: {e}')
m = gp.Model('Electricity_Procurement_Lot_Selection')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[option] * x_vars[option] for option in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[option] * x_vars[option] for option in option_ids)) >= 200, name='demand')
m.optimize()