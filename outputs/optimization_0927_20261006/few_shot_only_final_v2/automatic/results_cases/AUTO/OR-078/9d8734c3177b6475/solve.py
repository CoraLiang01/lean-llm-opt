import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = list(energy_df['option'])
required_cols = ['option', 'tech', 'gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Required column '{col}' not found in energy.csv")
gen_per_lot = {}
cost_per_lot = {}
tech = {}
for (idx, row) in energy_df.iterrows():
    option = str(row['option'])
    try:
        gen_per_lot[option] = float(row['gen_per_lot'])
    except Exception:
        raise ValueError(f"Invalid gen_per_lot for option '{option}': {row['gen_per_lot']}")
    try:
        cost_per_lot[option] = float(row['cost_per_lot'])
    except Exception:
        raise ValueError(f"Invalid cost_per_lot for option '{option}': {row['cost_per_lot']}")
    tech[option] = row['tech']
m = gp.Model('Electricity_Lot_Sizing')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[option] * x_vars[option] for option in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[option] * x_vars[option] for option in option_ids)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for option in option_ids:
        lots = x_vars[option].X
        if lots > 1e-06:
            print(f'Option: {option} | Tech: {tech[option]} | Lots: {int(round(lots))} | Gen per lot: {gen_per_lot[option]} | Cost per lot: {cost_per_lot[option]}')
    total_gen = sum((gen_per_lot[option] * x_vars[option].X for option in option_ids))
    print(f'Total generation: {total_gen:.2f} (Demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')