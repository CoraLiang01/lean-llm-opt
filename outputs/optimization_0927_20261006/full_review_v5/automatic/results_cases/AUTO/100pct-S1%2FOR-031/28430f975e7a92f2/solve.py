import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
if 'gen_per_lot' not in energy_df.columns:
    raise KeyError("Missing required column 'gen_per_lot' in energy.csv")
if 'cost_per_lot' not in energy_df.columns:
    raise KeyError("Missing required column 'cost_per_lot' in energy.csv")
if 'tech' not in energy_df.columns:
    raise KeyError("Missing required column 'tech' in energy.csv")
option_ids = energy_df['option'].tolist()
try:
    gen_per_lot = {row['option']: int(row['gen_per_lot']) for (_, row) in energy_df.iterrows()}
    cost_per_lot = {row['option']: float(row['cost_per_lot']) for (_, row) in energy_df.iterrows()}
    tech = {row['option']: row['tech'] for (_, row) in energy_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error converting numeric fields in energy.csv: {e}')
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= total_demand, name='DemandSatisfaction')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for opt in option_ids:
        lots = x_vars[opt].X
        if lots > 1e-06:
            print(f'  Option: {opt:15s} | Tech: {tech[opt]:12s} | Lots: {int(round(lots))} | Gen/lot: {gen_per_lot[opt]} | Cost/lot: {cost_per_lot[opt]:.2f}')
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (Demand: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')