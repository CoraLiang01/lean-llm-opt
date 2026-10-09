import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].astype(str).tolist()

def to_int(x):
    try:
        return int(x)
    except Exception:
        raise ValueError(f'Cannot convert to int: {x}')

def to_float(x):
    try:
        return float(x)
    except Exception:
        raise ValueError(f'Cannot convert to float: {x}')
gen_per_lot = {}
cost_per_lot = {}
for (idx, row) in energy_df.iterrows():
    option = str(row['option'])
    try:
        gen = to_int(row['gen_per_lot'])
        cost = to_float(row['cost_per_lot'])
    except Exception as e:
        raise ValueError(f"Error parsing numeric fields for option '{option}': {e}")
    gen_per_lot[option] = gen
    cost_per_lot[option] = cost
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[option] * x_vars[option] for option in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[option] * x_vars[option] for option in option_ids)) >= 200, name='demand')
m.optimize()