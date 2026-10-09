import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
try:
    cost_per_lot = {}
    gen_per_lot = {}
    tech = {}
    for (idx, row) in energy_df.iterrows():
        option = row['option']
        try:
            cost = float(row['cost_per_lot'])
        except Exception:
            raise ValueError(f"Missing or invalid cost_per_lot for option '{option}'")
        try:
            gen = int(row['gen_per_lot'])
        except Exception:
            raise ValueError(f"Missing or invalid gen_per_lot for option '{option}'")
        cost_per_lot[option] = cost
        gen_per_lot[option] = gen
        tech[option] = row['tech']
except KeyError as e:
    raise KeyError(f'Missing required column in energy.csv: {e}')
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= total_demand, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Option selections (number of lots purchased):')
    for opt in option_ids:
        val = x_vars[opt].X
        if val > 1e-06:
            print(f'  {opt}: {int(round(val))} lots (tech: {tech[opt]}, gen_per_lot: {gen_per_lot[opt]}, cost_per_lot: {cost_per_lot[opt]:.2f})')
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand required: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')