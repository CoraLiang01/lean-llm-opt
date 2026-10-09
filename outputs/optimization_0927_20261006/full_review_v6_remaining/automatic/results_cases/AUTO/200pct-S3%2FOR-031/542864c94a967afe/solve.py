import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
options = energy_df['option'].tolist()
if 'gen_per_lot' not in energy_df.columns:
    raise KeyError("Missing required column 'gen_per_lot' in energy.csv")
try:
    gen_per_lot = dict(zip(energy_df['option'], energy_df['gen_per_lot'].astype(int)))
except Exception as e:
    raise ValueError(f"Error converting 'gen_per_lot' to int: {e}")
if 'cost_per_lot' not in energy_df.columns:
    raise KeyError("Missing required column 'cost_per_lot' in energy.csv")
try:
    cost_per_lot = dict(zip(energy_df['option'], energy_df['cost_per_lot'].astype(float)))
except Exception as e:
    raise ValueError(f"Error converting 'cost_per_lot' to float: {e}")
if 'tech' not in energy_df.columns:
    raise KeyError("Missing required column 'tech' in energy.csv")
tech = dict(zip(energy_df['option'], energy_df['tech']))
for opt in options:
    if opt not in gen_per_lot or opt not in cost_per_lot or opt not in tech:
        raise ValueError(f"Missing required data for option '{opt}'.")
total_demand = 200
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in options)) >= total_demand, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected lots by contract option:')
    for opt in options:
        val = x_vars[opt].X
        if val >= 1e-06:
            print(f'  Option: {opt} | Tech: {tech[opt]} | Lots: {int(round(val))} | Gen per lot: {gen_per_lot[opt]} | Cost per lot: {cost_per_lot[opt]:.2f}')
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in options))
    print(f'Total generation purchased: {total_gen:.2f} (Demand: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')