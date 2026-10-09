import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
if len(option_ids) != len(energy_df):
    raise ValueError('Mismatch between option IDs and data rows in energy.csv.')
cost_per_lot = {}
gen_per_lot = {}
tech = {}
for (idx, row) in energy_df.iterrows():
    option = row['option']
    try:
        cost = float(row['cost_per_lot'])
        gen = int(row['gen_per_lot'])
        t = str(row['tech'])
    except Exception as e:
        raise ValueError(f"Error parsing numeric fields for option '{option}': {e}")
    cost_per_lot[option] = cost
    gen_per_lot[option] = gen
    tech[option] = t
demand = 200
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[option] * x_vars[option] for option in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[option] * x_vars[option] for option in option_ids)) >= demand, name='demand')
m.optimize()