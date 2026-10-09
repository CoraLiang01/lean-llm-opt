import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
if 'gen_per_lot' not in energy_df.columns:
    raise KeyError("Missing required column 'gen_per_lot' in energy.csv")
if 'cost_per_lot' not in energy_df.columns:
    raise KeyError("Missing required column 'cost_per_lot' in energy.csv")
option_ids = energy_df['option'].tolist()
try:
    gen_per_lot_dict = dict(zip(energy_df['option'], energy_df['gen_per_lot'].astype(float)))
    cost_per_lot_dict = dict(zip(energy_df['option'], energy_df['cost_per_lot'].astype(float)))
except Exception as e:
    raise ValueError(f'Error converting numeric fields in energy.csv: {e}')
if set(option_ids) != set(gen_per_lot_dict.keys()):
    raise ValueError('Mismatch between option_ids and gen_per_lot_dict keys')
if set(option_ids) != set(cost_per_lot_dict.keys()):
    raise ValueError('Mismatch between option_ids and cost_per_lot_dict keys')
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_dict[oid] * x_vars[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200.0
m.addConstr(gp.quicksum((gen_per_lot_dict[oid] * x_vars[oid] for oid in option_ids)) >= total_demand, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('--- Purchase Plan (lots per option) ---')
    for oid in option_ids:
        val = x_vars[oid].X
        if val > 1e-06:
            print(f'  {oid}: {int(round(val))} lots')
    total_gen = sum((gen_per_lot_dict[oid] * x_vars[oid].X for oid in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')