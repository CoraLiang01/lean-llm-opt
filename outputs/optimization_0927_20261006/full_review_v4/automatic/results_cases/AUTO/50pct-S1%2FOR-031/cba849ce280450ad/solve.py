import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
if len(option_ids) != len(set(option_ids)):
    raise ValueError('Duplicate option IDs found in energy.csv.')
try:
    gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(int).to_dict()
    cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
    tech = energy_df.set_index('option')['tech'].astype(str).to_dict()
except Exception as e:
    raise ValueError(f'Error converting parameter columns: {e}')
for oid in option_ids:
    if oid not in gen_per_lot or oid not in cost_per_lot or oid not in tech:
        raise KeyError(f'Missing parameter for option {oid}')
demand = 200
m = gp.Model('Electricity_Generation_Lot_Sizing')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[oid] * x_vars[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[oid] * x_vars[oid] for oid in option_ids)) >= demand, name='DemandSatisfaction')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected lots per option:')
    for oid in option_ids:
        val = x_vars[oid].X
        if val > 1e-06:
            print(f'  {oid}: {int(round(val))} lots (tech: {tech[oid]}, gen/lot: {gen_per_lot[oid]}, cost/lot: {cost_per_lot[oid]:.2f})')
    total_gen = sum((gen_per_lot[oid] * x_vars[oid].X for oid in option_ids))
    print(f'Total generation: {total_gen:.2f} (demand: {demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')